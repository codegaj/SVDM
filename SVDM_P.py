#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 16:45:33 2025

@author: ubuntu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from tqdm import tqdm
import cv2

from pytorch_msssim import ssim, SSIM
e = 1e-8
from models import Pannet, FusNet, PNNNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def vectorized_spatial_conv(x, kernels):
    """
    Vectorized spatial variant convolution
    x: [B, C, H, W]
    kernels: [B, H//4, W//4, C, Kh, Kw]
    """
    B, H, W, C = x.shape
    Kh, Kw = 15, 15
    pad = Kh // 2

    # Pad and extract patches
    x_padded = F.pad(x, [pad - 1, pad + 1, pad - 1, pad + 1])

    # Use unfold to extract patches
    patches = x_padded.unfold(2, Kh, 4).unfold(3, Kw, 4)  # [B, C, H//4, W//4, Kh, Kw]
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B, H//4, W//4, C, Kh, Kw]

    # Element-wise multiplication and sum
    output = torch.sum(patches * kernels, dim=[4, 5])

    return output.permute(0, 3, 1, 2)

class LocNet(nn.Module):
    """Location network"""

    def __init__(self, pe_dim=10):
        super(LocNet, self).__init__()
        self.pe_dim = pe_dim

        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU()
        )

        self.offset_head = nn.Linear(50, 2)
        self.spek_head = nn.Linear(50, 5)
        self.spak_head = nn.Linear(50, 4)

    def forward(self, coords):
        # coords: [B, N, 2] or [B*N, 2]
        original_shape = coords.shape
        if len(original_shape) == 3:
            coords = coords.reshape(-1, 2)

        features = self.net(coords)

        offset = self.offset_head(features)
        spek = self.spek_head(features)
        spak = self.spak_head(features)

        # Reshape back if needed
        if len(original_shape) == 3:
            batch_size, num_points, _ = original_shape
            offset = offset.reshape(batch_size, num_points, 2)
            spek = spek.reshape(batch_size, num_points, 5)
            spak = spak.reshape(batch_size, num_points, 4)

        return offset, spek, spak

class weighted_mae(nn.Module):
    def __init__(self,  reduction='mean'):

        super().__init__()
        self.reduction = reduction

    def _compute_weight_map(self, error_map):
        """计算权重图"""
        error_map = torch.clamp(error_map, min=0)
        weights = torch.log(self.a * error_map + 1) + 1

        return weights

    def forward(self, pred, target, error_map):

        self.mean_error = error_map.mean()
        self.a = 2/self.mean_error

        weight_map = self._compute_weight_map(error_map)

        # 计算绝对误差
        absolute_errors = torch.abs(pred - target)

        # 确保形状匹配
        if weight_map.dim() == 2 and pred.dim() > 2:
            # 扩展权重图
            weight_map = weight_map.unsqueeze(0).unsqueeze(0)
            weight_map = weight_map.expand(pred.shape[0], pred.shape[1], -1, -1)

        # 计算加权MAE
        weighted_errors = weight_map * absolute_errors

        # 归约
        if self.reduction == 'mean':
            return weighted_errors.mean()
        elif self.reduction == 'sum':
            return weighted_errors.sum()
        elif self.reduction == 'none':
            return weighted_errors

class SVDMModel(nn.Module):
    """Main SVDM model"""

    def __init__(self, ls=64, scale=4, ch_h=4, ch_m=1, pe_dim=10):
        super(SVDMModel, self).__init__()
        self.ls = ls
        self.scale = scale
        self.ch_h = ch_h
        self.ch_m = ch_m
        self.pe_dim = pe_dim

        self.loc_net = LocNet(pe_dim)

        # Learnable parameters
        self.sigma = nn.Parameter(torch.tensor([3.0]))
        self.half_k = nn.Parameter(torch.tensor(1.5))

    def compute_spatial_kernel(self, spak0):
        """Compute spatial kernel from location network output"""

        batch_size, _, __, ___ = spak0.shape

        # Reshape to [B, LS, LS, 2, 2]
        spak0 = spak0.reshape(-1, 2, 2)

        # Create mask and compute covariance matrix
        mask = torch.tensor([[1.0, 0], [1.0, 1.0]], device=spak0.device)
        mask = mask.unsqueeze(0).expand(spak0.shape[0], -1, -1)

        ms = spak0 * mask
        mst = ms.transpose(1, 2)

        INV = torch.bmm(mst, ms)

        # Create grid
        grid0 = torch.stack(torch.meshgrid(
            torch.arange(15, device=spak0.device),
            torch.arange(15, device=spak0.device),
            indexing='ij'
        ), dim=-1).float() - 7  # [15, 15, 2]

        gridt = grid0.unsqueeze(-1).permute(0, 1, 3, 2)

        gridt = gridt.reshape(1, 15 * 15, 2).expand(batch_size * self.ls * self.ls, -1, -1)

        # Compute Mahalanobis distance
        INV_flat = INV.reshape(-1, 2, 2)

        t = torch.bmm(gridt, INV_flat)  # [N, 1, 2]

        t = t.reshape(-1, 15, 15, 1, 2)

        gridl  = grid0.reshape(1, 15, 15, 2, 1).expand(batch_size * self.ls * self.ls, -1, -1, -1, -1)

        t = torch.matmul(t, gridl)  # [N, 1, 1]

        t = t.squeeze(-1).squeeze(-1)  # [N]

        t = t.reshape(batch_size, self.ls, self.ls, 15, 15)
        spak = torch.exp(-0.5 * t)  # [B, LS, LS, 15, 15]

        # Normalize
        sum_s = torch.sum(spak, dim=(3, 4), keepdim=True)
        spak = spak / sum_s

        return spak.unsqueeze(-1)  # [B, LS, LS, 15, 15, 1]

    def forward(self, coords, hrhs):
        batch_size = hrhs.shape[0]

        # Get offsets and parameters from location network
        offset, spek, spak0 = self.loc_net(coords)

        spak0 = spak0.reshape(batch_size, self.ls*4, self.ls*4, 4)
        spak0 = spak0[:, 1::4, 1::4, :]

        spek = spek.reshape(batch_size, self.ls*4, self.ls*4, 5)

        # Compute spatial kernel and apply to pan
        spak_kernel = self.compute_spatial_kernel(spak0)
        spak_kernel = spak_kernel.expand(-1, -1, -1, -1, -1, self.ch_h)
        spak_kernel = spak_kernel.permute(0, 1, 2, 5, 3, 4)  # [B, LS, LS, 1, 15, 15]
        # print(hrhs.shape, spak_kernel.shape)
        pre_msi = vectorized_spatial_conv(hrhs, spak_kernel)

        # Prepare for spectral transformation
        ones = torch.ones((batch_size, 1,  self.ls*4, self.ls*4), device=hrhs.device)
        new_hrhs = torch.cat([hrhs, ones], dim=1)

        # Reshape spectral parameters
        spek = spek.permute(0, 3, 1, 2)

        # Apply spectral transformation
        pre_pan = torch.sum(new_hrhs * spek, dim=1, keepdim=True)

        return  pre_msi, pre_pan

def load_and_preprocess_data():
    """Load and preprocess the data"""

    msi = np.load('./SVDM/reg_msi.npy')
    pan = np.load('./SVDM/ori_pan.npy')
    em = np.load('./SVDM/error_map.npy')

    msi = np.expand_dims(msi, 0)
    pan = np.expand_dims(pan, 0)

    em = np.expand_dims(em, 0)

    scale = 4
    new_M = min(int(pan.shape[1] / scale) * scale, int(msi.shape[1] * scale))
    new_N = min(int(pan.shape[2] / scale) * scale, int(msi.shape[2] * scale))

    pan = pan[:, :new_M, :new_N, :]
    msi = msi[:, :int(new_M / scale), :int(new_N / scale), :]

    print('pan shape: ', pan.shape, 'msi shape: ', msi.shape)

    # Create coordinate grid
    a = np.arange(pan.shape[1])
    b = np.arange(pan.shape[2])
    c = np.meshgrid(a, b, indexing='ij')

    c0 = np.expand_dims(-1 + (2 * (c[0]+20)) / (pan.shape[1]+40-1), -1)
    c1 = np.expand_dims(-1 + (2 * (c[1]+20)) / (pan.shape[2]+40-1), -1)
    ce = np.concatenate((c0, c1), -1)
    ce = np.expand_dims(ce, 0)

    return msi, pan, em, ce, scale

def create_training_patches(msi, pan, em, ce, ls=64, scale=4, stride=256):
    """Create training patches"""
    hs = scale * ls
    new_M, new_N = pan.shape[1], pan.shape[2]

    train_msi_all = []
    train_pan_all = []
    train_coor = []

    train_error = []

    for j in tqdm(range(0, new_M - hs + 1, stride)):
        if j + hs > new_M:
            start_j = new_M - hs
        else:
            start_j = j

        for k in range(0, new_N - hs + 1, stride):
            if k + hs > new_N:
                start_k = new_N - hs
            else:
                start_k = k

            temp_pan = pan[0, start_j:start_j + hs, start_k:start_k + hs, :]
            temp_msi = msi[0, int(start_j / scale):int(start_j / scale) + ls,
                        int(start_k / scale):int(start_k / scale) + ls, :]
            temp_coor = ce[0, start_j:start_j + hs, start_k:start_k + hs, :]

            temp_error = em[0, int(start_j / scale):int(start_j / scale) + ls,
                        int(start_k / scale):int(start_k / scale) + ls, :]

            temp1 = cv2.blur(np.float64(temp_pan[:, :, 0]), (5, 5))
            if np.min(temp1) > 0:

                train_msi_all.append(temp_msi)
                train_pan_all.append(temp_pan)
                train_coor.append(temp_coor)
                train_error.append(temp_error)

    train_msi_all = np.array(train_msi_all, dtype='float32')
    train_pan_all = np.array(train_pan_all, dtype='float32')
    train_coor = np.array(train_coor, dtype='float32')
    train_coor = np.reshape(train_coor, (-1, hs * hs, 2))

    train_error = np.array(train_error, dtype='float32')

    return train_msi_all, train_pan_all, train_error, train_coor

def train_model():
    """Main training function"""
    # Load data
    msi, pan, em, ce, scale = load_and_preprocess_data()
    train_msi_all, train_pan_all, train_error, train_coor = create_training_patches(msi, pan, em, ce)

    # Initialize model
    model = SVDMModel().to(device)
    model.load_state_dict(torch.load('./SVDM/GF7_model.pth'))

    # fusion_model = Pannet().to(device)
    # fusion_model = FusNet().to(device)
    fusion_model = PNNNet().to(device)

    # 冻结 model_a 的参数
    for param in model.parameters():
        param.requires_grad = False

    optimizer2 = Adam(fusion_model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    optimizer3 = Adam(fusion_model.parameters(), lr=0.00005, betas=(0.9, 0.999))

    # Training loop
    batch_size = 64
    epochs = 50

    ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)
    mae_loss = weighted_mae()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        # Shuffle indices
        indices = torch.randperm(train_msi_all.shape[0])

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]

            batch_msi = train_msi_all[batch_indices]
            batch_pan = train_pan_all[batch_indices]
            batch_coor = train_coor[batch_indices]
            batch_error = train_error[batch_indices]

            # Convert to PyTorch tensors
            batch_msi = torch.from_numpy(batch_msi).to(device)
            batch_pan = torch.from_numpy(batch_pan).to(device)
            batch_coor = torch.from_numpy(batch_coor).to(device)
            batch_error = torch.from_numpy(batch_error).to(device)

            batch_msi = batch_msi.permute(0,3,1,2)
            batch_pan = batch_pan.permute(0,3,1,2)
            batch_error = batch_error.permute(0,3,1,2)

            # Select optimizer based on epoch
            if epoch < 40:
                optimizer = optimizer2
            else:
                optimizer = optimizer3

            optimizer.zero_grad()

            # Forward pass
            o_hrhs = fusion_model(batch_msi, batch_pan)
            pre_msi, pre_pan = model(batch_coor, o_hrhs)

            spa_loss = 1 - ssim_loss(batch_pan, pre_pan)
            spe_loss = mae_loss(batch_msi, pre_msi, batch_error)

            loss = spa_loss+spe_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if i % (10 * batch_size) == 0:
                print(f'Batch {i // batch_size}: loss={loss:.4f}')

        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}')

        # 保存模型和结果
        if epoch == epochs - 1:
            torch.save(fusion_model.state_dict(), './fus_results/GF7_fusion_model.pth')
            print("Model saved successfully!")

    return model

def eval_model():
    msi, pan, em, ce, scale = load_and_preprocess_data()

    # Initialize model
    model = SVDMModel().to(device)
    model.load_state_dict(torch.load('./SVDM/GF7/GF7_model.pth'))

    # fusion_model = Pannet().to(device)
    # fusion_model = FusNet().to(device)
    fusion_model = PNNNet().to(device)
    fusion_model.load_state_dict(torch.load('./fus_results/GF7_fusion_model.pth'))

    with torch.no_grad():

        ls = 64
        hs = ls*scale

        ch_m = 1
        ch_h = 4

        new_M = pan.shape[1]
        new_N = pan.shape[2]

        rec_size = (ls - 10) * scale

        rec_hs = new_M - 10 * scale
        rec_ws = new_N - 10 * scale

        fus_msi = np.zeros((1, rec_hs, rec_ws, ch_h), dtype='uint8')
        ori_pan = np.zeros((1, rec_hs, rec_ws, 1), dtype='uint8')
        reg_msi = np.zeros((1, int(rec_hs / 4), int(rec_ws / 4), ch_h), dtype='uint8')
        pre_msi = np.zeros((1, int(rec_hs / 4), int(rec_ws / 4), ch_h), dtype='uint8')

        #here register msi image
        for ji, j in enumerate(tqdm(range(0, new_M - hs + 1 + rec_size, rec_size))):

            if j + hs > new_M:
                start_j = new_M - hs
            else:
                start_j = j

            for ki, k in enumerate(range(0, new_N - hs + 1 + rec_size, rec_size)):

                if k + hs > new_N:
                    start_k = new_N - hs
                else:
                    start_k = k

                temp_pan = pan[:, start_j:start_j + hs, start_k:start_k + hs, :]
                temp_msi = msi[:, int(start_j / scale):int(start_j / scale) + ls,
                            int(start_k / scale):int(start_k / scale) + ls, :]

                temp1 = cv2.blur(np.float64(temp_pan[:, :, 0]), (5, 5))
                if np.min(temp1) > 0:

                    temp_coor = ce[:, start_j:start_j + hs, start_k:start_k + hs, :]
                    temp_coor = np.reshape(temp_coor, (1, hs * hs, 2))

                    ori_pan[:, start_j:start_j + rec_size, start_k:start_k + rec_size, :] = np.uint8(np.clip(temp_pan[:,
                                                                                            5 * scale:-5 * scale,
                                                                                            5 * scale:-5 * scale, :], 0, 1)*255)
                    reg_msi[:, int(start_j / scale):int(start_j / scale) + int(rec_size / scale),
                    int(start_k / scale):int(start_k / scale) + int(rec_size / scale), :] = np.uint8(np.clip(temp_msi[:, 5:-5, 5:-5, :], 0, 1)*255)

                    temp_msi = torch.from_numpy(np.float32(temp_msi)).to(device)
                    temp_pan = torch.from_numpy(np.float32(temp_pan)).to(device)
                    temp_coor = torch.from_numpy(np.float32(temp_coor)).to(device)

                    temp_msi = temp_msi.permute(0, 3, 1, 2)
                    temp_pan = temp_pan.permute(0, 3, 1, 2)

                    o_hrhs = fusion_model(temp_msi, temp_pan)
                    d_msi, d_pan = model(temp_coor, o_hrhs)

                    o_hrhs = o_hrhs.permute(0, 2, 3, 1)
                    d_pan = d_pan.permute(0, 2, 3, 1)
                    d_msi = d_msi.permute(0, 2, 3, 1)

                    o_hrhs = o_hrhs[:, 5 * scale:-5 * scale, 5 * scale:-5 * scale, :]
                    d_msi = d_msi[:, 5:-5, 5:-5, :]

                    fus_msi[:, start_j:start_j + rec_size, start_k:start_k + rec_size, :] = np.uint8(np.clip(o_hrhs.cpu().numpy(), 0, 1)*255)

                    pre_msi[:, int(start_j / scale):int(start_j / scale) + int(rec_size / scale),
                    int(start_k / scale):int(start_k / scale) + int(rec_size / scale), :] = np.uint8(np.clip(d_msi.cpu().numpy(), 0, 1)*255)

        np.save('./fus_results/rec_msi.npy', fus_msi)
        np.save('./fus_results/ori_pan.npy', ori_pan)
        np.save('./fus_results/reg_msi.npy', reg_msi)
        np.save('./fus_results/pre_msi.npy', pre_msi)

if __name__ == "__main__":
    model = train_model()
    eval_model()