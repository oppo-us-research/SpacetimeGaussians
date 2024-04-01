# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.nn as nn

from mmcv.ops import knn
from simple_knn._C import distCUDA2

from thirdparty.gaussian_splatting.utils.graphics_utils import BasicPointCloud


class Sandwich(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwich, self).__init__()

        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias)  #

        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3, dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1)  # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = self.sigmoid(result)
        return result


class Sandwichnoact(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwichnoact, self).__init__()

        # self.mlp2 = nn.Conv2d(11, 6, kernel_size=1, bias=bias) # double hidden layer..
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias)  # double hidden layer..

        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3, dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1)  # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = torch.clamp(result, min=0.0, max=1.0)
        return result


####### following are also good rgb model but not used in the paper, slower than sandwich, inspired by color shift in hyperreel
# remove sigmoid for immersive dataset
class RGBDecoderVRayShift(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(RGBDecoderVRayShift, self).__init__()

        self.mlp1 = nn.Conv2d(dim, outdim, kernel_size=1, bias=bias)
        self.mlp2 = nn.Conv2d(15, outdim, kernel_size=1, bias=bias)
        self.mlp3 = nn.Conv2d(6, outdim, kernel_size=1, bias=bias)
        self.sigmoid = torch.nn.Sigmoid()

        self.dwconv1 = nn.Conv2d(9, 9, kernel_size=1, bias=bias)

    def forward(self, input, rays, t=None):
        x = self.dwconv1(input) + input
        albeado = self.mlp1(x)
        specualr = torch.cat([x, rays], dim=1)
        specualr = self.mlp2(specualr)

        finalfeature = torch.cat([albeado, specualr], dim=1)
        result = self.mlp3(finalfeature)
        result = self.sigmoid(result)
        return result


def interpolate_point(pcd, N=4):

    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times

    timestamps = np.unique(oldtime)

    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selected_mask = oldtime == time
        selected_mask = selected_mask.squeeze(1)

        if timeidx == 0:
            newxyz.append(oldxyz[selected_mask])
            newcolor.append(oldcolor[selected_mask])
            newnormal.append(oldnormal[selected_mask])
            newtime.append(oldtime[selected_mask])
        else:
            xyzinput = oldxyz[selected_mask]
            xyzinput = torch.from_numpy(xyzinput).float().cuda()
            xyzinput = xyzinput.unsqueeze(0).contiguous()  # 1 x N x 3
            xyznnpoints = knn(2, xyzinput, xyzinput, False)

            nearestneibourindx = xyznnpoints[0, 1].long()  # N x 1
            spatialdistance = torch.norm(xyzinput - xyzinput[:, nearestneibourindx, :], dim=2)  #  1 x N
            spatialdistance = spatialdistance.squeeze(0)

            diff_sorted, _ = torch.sort(spatialdistance)
            N = spatialdistance.shape[0]
            num_take = int(N * 0.25)
            masks = spatialdistance > diff_sorted[-num_take]
            masksnumpy = masks.cpu().numpy()

            newxyz.append(oldxyz[selected_mask][masksnumpy])
            newcolor.append(oldcolor[selected_mask][masksnumpy])
            newnormal.append(oldnormal[selected_mask][masksnumpy])
            newtime.append(oldtime[selected_mask][masksnumpy])
            #
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]

    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd


def interpolate_point_v3(pcd, N=4, m=0.25):

    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times

    timestamps = np.unique(oldtime)

    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selected_mask = oldtime == time
        selected_mask = selected_mask.squeeze(1)

        if timeidx % N == 0:
            newxyz.append(oldxyz[selected_mask])
            newcolor.append(oldcolor[selected_mask])
            newnormal.append(oldnormal[selected_mask])
            newtime.append(oldtime[selected_mask])

        else:
            xyzinput = oldxyz[selected_mask]
            xyzinput = torch.from_numpy(xyzinput).float().cuda()
            xyzinput = xyzinput.unsqueeze(0).contiguous()  # 1 x N x 3
            xyznnpoints = knn(2, xyzinput, xyzinput, False)

            nearestneibourindx = xyznnpoints[
                0, 1
            ].long()  # N x 1  skip the first one, we select the second closest one
            spatialdistance = torch.norm(xyzinput - xyzinput[:, nearestneibourindx, :], dim=2)  #  1 x N
            spatialdistance = spatialdistance.squeeze(0)

            diff_sorted, _ = torch.sort(spatialdistance)
            M = spatialdistance.shape[0]
            num_take = int(M * m)
            masks = spatialdistance > diff_sorted[-num_take]
            masksnumpy = masks.cpu().numpy()

            newxyz.append(oldxyz[selected_mask][masksnumpy])
            newcolor.append(oldcolor[selected_mask][masksnumpy])
            newnormal.append(oldnormal[selected_mask][masksnumpy])
            newtime.append(oldtime[selected_mask][masksnumpy])
            #
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]

    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd


def interpolate_part_use(pcd, N=4):
    # used in ablation study
    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times

    timestamps = np.unique(oldtime)

    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selected_mask = oldtime == time
        selected_mask = selected_mask.squeeze(1)

        if timeidx % N == 0:
            newxyz.append(oldxyz[selected_mask])
            newcolor.append(oldcolor[selected_mask])
            newnormal.append(oldnormal[selected_mask])
            newtime.append(oldtime[selected_mask])

        else:
            pass
            #
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]

    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd


def padding_point(pcd, N=4):

    oldxyz = pcd.points
    oldcolor = pcd.colors
    oldnormal = pcd.normals
    oldtime = pcd.times

    timestamps = np.unique(oldtime)
    totallength = len(timestamps)

    newxyz = []
    newcolor = []
    newnormal = []
    newtime = []
    for timeidx, time in enumerate(timestamps):
        selected_mask = oldtime == time
        selected_mask = selected_mask.squeeze(1)

        if timeidx != 0 and timeidx != len(timestamps) - 1:
            newxyz.append(oldxyz[selected_mask])
            newcolor.append(oldcolor[selected_mask])
            newnormal.append(oldnormal[selected_mask])
            newtime.append(oldtime[selected_mask])

        else:
            newxyz.append(oldxyz[selected_mask])
            newcolor.append(oldcolor[selected_mask])
            newnormal.append(oldnormal[selected_mask])
            newtime.append(oldtime[selected_mask])

            xyzinput = oldxyz[selected_mask]
            xyzinput = torch.from_numpy(xyzinput).float().cuda()
            xyzinput = xyzinput.unsqueeze(0).contiguous()  # 1 x N x 3

            xyznnpoints = knn(2, xyzinput, xyzinput, False)

            nearestneibourindx = xyznnpoints[
                0, 1
            ].long()  # N x 1  skip the first one, we select the second closest one
            spatialdistance = torch.norm(xyzinput - xyzinput[:, nearestneibourindx, :], dim=2)  #  1 x N
            spatialdistance = spatialdistance.squeeze(0)

            diff_sorted, _ = torch.sort(spatialdistance)
            N = spatialdistance.shape[0]
            num_take = int(N * 0.125)
            masks = spatialdistance > diff_sorted[-num_take]
            masksnumpy = masks.cpu().numpy()

            newxyz.append(oldxyz[selected_mask][masksnumpy])
            newcolor.append(oldcolor[selected_mask][masksnumpy])
            newnormal.append(oldnormal[selected_mask][masksnumpy])

            if timeidx == 0:
                newtime.append(oldtime[selected_mask][masksnumpy] - (1 / totallength))
            else:
                newtime.append(oldtime[selected_mask][masksnumpy] + (1 / totallength))
    newxyz = np.concatenate(newxyz, axis=0)
    newcolor = np.concatenate(newcolor, axis=0)
    newtime = np.concatenate(newtime, axis=0)
    assert newxyz.shape[0] == newcolor.shape[0]

    newpcd = BasicPointCloud(points=newxyz, colors=newcolor, normals=None, times=newtime)

    return newpcd


def get_color_model(rgb_function):
    if rgb_function == "sandwich":
        rgb_decoder = Sandwich(9, 3)

    elif rgb_function == "sandwichnoact":
        rgb_decoder = Sandwichnoact(9, 3)
    else:
        return None
    return rgb_decoder


def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0


def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5
