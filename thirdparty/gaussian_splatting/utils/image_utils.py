#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def psnr_mask(img1, img2):
    # img1 = img1.squeeze(0)
    # img2 = img2.squeeze(0)
    mask = img2 > 0.0
    valid_mask = torch.sum(img2[:, :, :], dim=1) > 0.01
    valid_mask = valid_mask.repeat(3, 1, 1)  # .float()
    valid_mask = valid_mask.view(img1.shape[0], -1)

    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1)[:, valid_mask[0, :]].mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# def psnr_mask(img, img2, mask):
#     mse = (((img - mask)) ** 2).view(img.shape[0], -1).mean(1, keepdim=True)
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))
