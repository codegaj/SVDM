The official implementation of paper "Spatially-Varying Degradation Modeling and Pansharpening for Large-Scale Multispectral Images"

#Usage

(1) Download the dataset from https://pan.baidu.com/s/1AljE541KPEPld8XeP-hTqw?pwd=72hx, and unzip them to "./datasets/".

(2) Run SVDM.py, then you can get the registered MSI, PAN, and the trained SVDMNet in "./SVDM/".

(3) Run SVDM_P.py, then you can get the pansharpened HR-MSI in "./fus_results/".

#Note

(1) If you have downloading troubles about the above-mentioned links, please email me: anjing_guo@hnu.edu.cn.

(2) We appreciate the original providers of the datasets, and the above datasets can only be used for academic purposes.

#Device

Nvidia RTX4090 GPU + 128GB RAM

#Enviroments

ubuntu 20.04 + cuda 12.6 + python 3.10 +pytorch 2.5.1