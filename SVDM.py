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
import cv2
import scipy.io as sio
import time
import random
from torch.optim import Adam, SGD
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from pytorch_msssim import ssim, SSIM
e = 1e-8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OffsetLayer(nn.Module):

    def __init__(self):
        super(OffsetLayer, self).__init__()

    def forward(self, inputs):
        offsets, x = inputs

        batch_size, channels, height, width  = x.shape

        _, N, __ = offsets.shape

        offsets = offsets.reshape(-1, int(np.sqrt(N)), int(np.sqrt(N)), 2)

        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device),
            torch.linspace(-1, 1, width, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, H, W, 2]

        # Apply offsets (normalize offsets to [-1, 1] range)
        offsets_normalized = offsets / torch.tensor([width, height], device=x.device).view(1, 1, 1, 2) * 2
        new_grid = grid + offsets_normalized

        # Sample using grid_sample
        x_offset = F.grid_sample(x, new_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return x_offset

def vectorized_spatial_conv(x, kernels):
    """
    Vectorized spatial variant convolution
    x: [B, H, W, C]
    kernels: [B, H//4, W//4, C, Kh, Kw]
    """
    B, C, H, W = x.shape
    Kh, Kw = 15, 15
    pad = Kh // 2

    # Pad and extract patches, start point 1
    x_padded = F.pad(x, [pad - 1, pad + 1, pad - 1, pad + 1])

    # Use unfold to extract patches
    patches = x_padded.unfold(2, Kh, 4).unfold(3, Kw, 4)  # [B, C, H//4, W//4, Kh, Kw]
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B, H//4, W//4, C, Kh, Kw]

    # Element-wise multiplication and sum
    output = torch.sum(patches * kernels, dim=[3, 4, 5])

    return output.unsqueeze(1)  # [B, 1, H//4, W//4]

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

class SVDMModel(nn.Module):

    def __init__(self, ls=64, scale=4, ch_h=4, ch_m=1, pe_dim=10):
        super(SVDMModel, self).__init__()
        self.ls = ls
        self.scale = scale
        self.ch_h = ch_h
        self.ch_m = ch_m
        self.pe_dim = pe_dim

        self.loc_net = LocNet(pe_dim)
        self.offset_layer = OffsetLayer()

        # Learnable parameters
        self.sigma = nn.Parameter(torch.tensor([3.0]))
        self.half_k = nn.Parameter(torch.tensor(1.5))

    def compute_spatial_kernel(self, spak0):

        batch_size, num_points, _ = spak0.shape
        spak0 = spak0.reshape(batch_size, self.ls, self.ls, 4)

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

    def forward(self, msi, pan, coords):
        batch_size = msi.shape[0]

        # Get offsets and parameters from location network
        offset, spek, spak0 = self.loc_net(coords)

        # Apply offset to msi
        o_msi = self.offset_layer([offset, msi])
        o_msi = o_msi[:, :, 5:-5, 5:-5]  # Remove borders

        # Compute spatial kernel and apply to pan
        spak_kernel = self.compute_spatial_kernel(spak0)
        spak_kernel = spak_kernel.permute(0, 1, 2, 5, 3, 4)  # [B, LS, LS, 1, 15, 15]
        pre_pan = vectorized_spatial_conv(pan, spak_kernel)
        pre_pan = pre_pan[:, :, 5:-5, 5:-5]  # Remove borders

        # Prepare for spectral transformation
        ones = torch.ones((batch_size, 1,  self.ls - 10, self.ls - 10), device=msi.device)
        o_msi1 = torch.cat([o_msi, ones], dim=1)

        o_msi1 = o_msi1.permute(0, 2, 3, 1)

        # Reshape spectral parameters
        spek = spek.reshape(batch_size, self.ls, self.ls, 5)
        spek = spek[:, 5:-5, 5:-5, :]

        # Apply spectral transformation
        pre_msi = torch.sum(o_msi1 * spek, dim=-1, keepdim=True)

        pre_msi = pre_msi.permute(0, 3, 1, 2)

        o_offset = offset.reshape(-1, self.ls, self.ls, 2)
        o_offset = o_offset[:, 5:-5, 5:-5, :]

        return o_msi, o_offset, pre_msi, pre_pan, offset, spek, spak0

# Data loading and preprocessing (same as original)
def load_and_preprocess_data():
    """Load and preprocess the data"""
    msi = np.load('./datasets/over_msi_gf7.npy')
    msi = msi / np.max(msi)
    pan = np.load('./datasets/over_pan_gf7.npy')
    pan = pan / np.max(pan)

    msi = np.expand_dims(msi, 0)
    pan = np.expand_dims(pan, 0)

    scale = 4
    new_M = min(int(pan.shape[1] / scale) * scale, int(msi.shape[1] * scale))
    new_N = min(int(pan.shape[2] / scale) * scale, int(msi.shape[2] * scale))

    pan = pan[:, :new_M, :new_N, :]
    msi = msi[:, :int(new_M / scale), :int(new_N / scale), :]

    print('pan shape: ', pan.shape, 'msi shape: ', msi.shape)

    # Create coordinate grid
    a = np.arange(msi.shape[1])
    b = np.arange(msi.shape[2])
    c = np.meshgrid(a, b, indexing='ij')

    c0 = np.expand_dims(-1 + (2 * c[0]) / (msi.shape[1]-1), -1)
    c1 = np.expand_dims(-1 + (2 * c[1]) / (msi.shape[2]-1), -1)
    ce = np.concatenate((c0, c1), -1)
    ce = np.expand_dims(ce, 0)

    return msi, pan, ce, scale

def create_training_patches(msi, pan, ce, ls=64, scale=4, stride=256):
    hs = scale * ls
    new_M, new_N = pan.shape[1], pan.shape[2]

    train_msi_all = []
    train_pan_all = []
    train_coor = []

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
            temp_coor = ce[0, int(start_j / scale):int(start_j / scale) + ls,
                        int(start_k / scale):int(start_k / scale) + ls, :]

            if np.min(temp_pan) > 0 and np.min(temp_msi) > 0:
                train_msi_all.append(temp_msi)
                train_pan_all.append(temp_pan)
                train_coor.append(temp_coor)

    train_msi_all = np.array(train_msi_all, dtype='float32')
    train_pan_all = np.array(train_pan_all, dtype='float32')
    train_coor = np.array(train_coor, dtype='float32')
    train_coor = np.reshape(train_coor, (-1, ls * ls, 2))

    return train_msi_all, train_pan_all, train_coor

def train_model():
    # Load data
    msi, pan, ce, scale = load_and_preprocess_data()
    train_msi_all, train_pan_all, train_coor = create_training_patches(msi, pan, ce)

    # Convert to PyTorch tensors
    train_msi_all = torch.from_numpy(train_msi_all).to(device)
    train_pan_all = torch.from_numpy(train_pan_all).to(device)
    train_coor = torch.from_numpy(train_coor).to(device)

    # Initialize model
    model = SVDMModel().to(device)

    optimizer2 = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    optimizer3 = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

    # Training loop
    batch_size = 64
    epochs = 50

    ssim_module = SSIM(data_range=1.0, size_average=True, channel=1)  # channel: 图像通道数

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

            batch_msi = batch_msi.permute(0, 3 ,1, 2)
            batch_pan = batch_pan.permute(0, 3 ,1, 2)

            # Select optimizer based on epoch
            if epoch < 40:
                optimizer = optimizer2
            else:
                optimizer = optimizer3

            optimizer.zero_grad()

            # Forward pass
            o_msi, o_offset, d_msi, d_pan, offset, spek, spak0 = model(batch_msi, batch_pan, batch_coor)

            # Compute loss (SSIM-like loss)
            ssim_loss = 1 - ssim_module(d_msi, d_pan)

            ssim_loss.backward()
            optimizer.step()

            total_loss += ssim_loss.item()
            num_batches += 1

            if i % (10 * batch_size) == 0:
                print(f'Batch {i // batch_size}: loss={ssim_loss:.4f}')

        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}')

        # 保存模型和结果
        if epoch == epochs - 1:
            torch.save(model.state_dict(), './SVDM/GF7_model.pth')
            print("Model saved successfully!")

    return model

def eval_model():
    msi, pan, ce, scale = load_and_preprocess_data()

    # Initialize model
    model = SVDMModel().to(device)
    model.load_state_dict(torch.load('./SVDM/GF7_model.pth'))

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

        rec_pan = np.zeros((1, rec_hs, rec_ws, ch_m), dtype='float16')
        rec_msi = np.zeros((1, int(rec_hs / scale), int(rec_ws / scale), ch_h), dtype='float16')
        rec_off = np.zeros((1, int(rec_hs / scale), int(rec_ws / scale), 2), dtype='float16')
        ori_msi = np.zeros((1, int(rec_hs / scale), int(rec_ws / scale), ch_h), dtype='float16')

        used_pan = np.zeros((1, int(rec_hs / scale), int(rec_ws / scale), ch_m), dtype='float16')
        used_msi = np.zeros((1, int(rec_hs / scale), int(rec_ws / scale), ch_m), dtype='float16')

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

                if np.min(temp_pan) > 0 and np.min(temp_msi) > 0:
                    temp_coor = ce[:, int(start_j / scale):int(start_j / scale) + ls,
                                int(start_k / scale):int(start_k / scale) + ls, :]
                    temp_coor = np.reshape(temp_coor, (1, ls * ls, 2))

                    rec_pan[:, start_j:start_j + rec_size, start_k:start_k + rec_size, :] = temp_pan[:,
                                                                                            5 * scale:-5 * scale,
                                                                                            5 * scale:-5 * scale, :]
                    ori_msi[:, int(start_j / scale):int(start_j / scale) + int(rec_size / scale),
                    int(start_k / scale):int(start_k / scale) + int(rec_size / scale), :] = temp_msi[:, 5:-5, 5:-5, :]

                    temp_msi = torch.from_numpy(np.float32(temp_msi)).to(device)
                    temp_pan = torch.from_numpy(np.float32(temp_pan)).to(device)
                    temp_coor = torch.from_numpy(np.float32(temp_coor)).to(device)

                    temp_pan = temp_pan.permute(0, 3, 1, 2)
                    temp_msi = temp_msi.permute(0, 3, 1, 2)

                    o_msi, o_offset, pre_msi, pre_pan, offset, spek, spak0 = model(temp_msi, temp_pan,
                                                                                      temp_coor)

                    pre_msi = pre_msi.permute(0, 2, 3, 1)
                    pre_pan = pre_pan.permute(0, 2, 3, 1)
                    o_msi = o_msi.permute(0, 2, 3, 1)

                    rec_msi[:, int(start_j / scale):int(start_j / scale) + int(rec_size / scale),
                    int(start_k / scale):int(start_k / scale) + int(rec_size / scale), :] = o_msi.cpu().numpy()
                    rec_off[:, int(start_j / scale):int(start_j / scale) + int(rec_size / scale),
                    int(start_k / scale):int(start_k / scale) + int(rec_size / scale), :] = o_offset.cpu().numpy()
                    used_pan[:, int(start_j / scale):int(start_j / scale) + int(rec_size / scale),
                    int(start_k / scale):int(start_k / scale) + int(rec_size / scale), :] = pre_pan.cpu().numpy()
                    used_msi[:, int(start_j / scale):int(start_j / scale) + int(rec_size / scale),
                    int(start_k / scale):int(start_k / scale) + int(rec_size / scale), :] = pre_msi.cpu().numpy()

        plt.imsave('./SVDM/x_shift.png', rec_off[0, :, :, 0], cmap='jet')
        plt.imsave('./SVDM/y_shift.png', rec_off[0, :, :, 1], cmap='jet')

        error = np.abs(used_pan-used_msi)
        #
        np.save('./SVDM/ori_pan', rec_pan[0, :, :, :])
        np.save('./SVDM/reg_msi', rec_msi[0, :, :, :])
        np.save('./SVDM/ori_msi', ori_msi[0, :, :, :])
        np.save('./SVDM/rec_off', rec_off[0, :, :, :])

        np.save('./SVDM/error_map', error[0, :, :, :])

if __name__ == "__main__":
    model = train_model()
    eval_model()