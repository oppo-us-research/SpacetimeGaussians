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

import os

import numpy as np
import torch

from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from helper_model import (
    get_color_model,
    interpolate_part_use,
    interpolate_point,
    interpolate_point_v3,
)
from thirdparty.gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
    update_quaternion,
)
from thirdparty.gaussian_splatting.utils.graphics_utils import BasicPointCloud
from thirdparty.gaussian_splatting.utils.sh_utils import rgb2sh
from thirdparty.gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp  # special set for visual examples
        self.scaling_inverse_activation = torch.log  # special set for vislual examples

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.feature_act = torch.sigmoid

    def __init__(self, sh_degree: int, rgb_function="rgbv1"):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._motion = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._omega = torch.empty(0)

        self.rgb_decoder = get_color_model(rgb_function)

        self.setup_functions()
        self.delta_t = None
        self.omega_mask = None
        self.mask_for_ems = None
        self.distance_to_camera = None
        self.trbf_scale_init = None
        self.ts = None
        self.trbf_output = None
        self.preprocess_points = False
        self.add_sph_points_scale = 0.8

        self.max_z, self.min_z = 0.0, 0.0
        self.max_y, self.min_y = 0.0, 0.0
        self.max_x, self.min_x = 0.0, 0.0
        self.ray_start = 0.7
        self.computed_trbf_scale = None
        self.computed_opacity = None
        self.computed_scales = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    def get_rotation(self, delta_t):
        rotation = self._rotation + delta_t * self._omega
        self.delta_t = delta_t
        return self.rotation_activation(rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_trbf_center(self):
        return self._trbf_center

    @property
    def get_trbf_scale(self):
        return self._trbf_scale

    def get_features(self, delta_t):
        return torch.cat((self._features_dc, delta_t * self._features_t), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def one_up_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):

        if self.preprocess_points == 3:
            pcd = interpolate_point(pcd, 4)

        elif self.preprocess_points == 4:
            pcd = interpolate_point(pcd, 2)

        elif self.preprocess_points == 5:
            pcd = interpolate_point(pcd, 6)

        elif self.preprocess_points == 6:
            pcd = interpolate_point(pcd, 8)

        elif self.preprocess_points == 7:
            pcd = interpolate_point(pcd, 16)

        elif self.preprocess_points == 8:
            pcd = interpolate_point_v3(pcd, 4)

        elif self.preprocess_points == 14:
            pcd = interpolate_part_use(pcd, 2)

        elif self.preprocess_points == 15:
            pcd = interpolate_part_use(pcd, 4)

        elif self.preprocess_points == 16:
            pcd = interpolate_part_use(pcd, 8)

        elif self.preprocess_points == 17:
            pcd = interpolate_part_use(pcd, 16)

        else:
            pass

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        times = torch.tensor(np.asarray(pcd.times)).float().cuda()

        print("Number of points at initialization : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        features9channel = torch.cat((fused_color, fused_color), dim=1)

        self._features_dc = nn.Parameter(features9channel.contiguous().requires_grad_(True))

        N, _ = fused_color.shape

        f_omega = torch.zeros((N, 3), dtype=torch.float, device="cuda")
        self._features_t = nn.Parameter(f_omega.contiguous().requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))

        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        motion = torch.zeros((fused_point_cloud.shape[0], 9), device="cuda")  # x1, x2, x3,  y1,y2,y3, z1,z2,z3
        self._motion = nn.Parameter(motion.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True))

        ## store gradients

        if self.trbf_scale_init is not None:
            nn.init.constant_(self._trbf_scale, self.trbf_scale_init)  # too large ?
        else:
            nn.init.constant_(self._trbf_scale, 0)  # too large ?

        nn.init.constant_(self._features_t, 0)
        nn.init.constant_(self._omega, 0)
        self.rgb_grd = {}

        self.max_z, self.min_z = torch.amax(self._xyz[:, 2]), torch.amin(self._xyz[:, 2])
        self.max_y, self.min_y = torch.amax(self._xyz[:, 1]), torch.amin(self._xyz[:, 1])
        self.max_x, self.min_x = torch.amax(self._xyz[:, 0]), torch.amin(self._xyz[:, 0])
        self.max_z = min((self.max_z, 200.0))  # some outliers in the n4d datasets..

        for name, W in self.rgb_decoder.named_parameters():
            if "weight" in name:
                self.rgb_grd[name] = torch.zeros_like(W, requires_grad=False).cuda()
                # self.rgb_grd[name] + W.grad.clone()
            elif "bias" in name:
                print("not implemented")
                quit()

    def cache_gradient(self):
        self._xyz_grd += self._xyz.grad.clone()
        self._features_dc_grd += self._features_dc.grad.clone()
        self._features_t_grd += self._features_t.grad.clone()  # self._features_t_grd
        self._scaling_grd += self._scaling.grad.clone()
        self._rotation_grd += self._rotation.grad.clone()
        self._opacity_grd += self._opacity.grad.clone()
        self._trbf_center_grd += self._trbf_center.grad.clone()
        self._trbf_scale_grd += self._trbf_scale.grad.clone()
        self._motion_grd += self._motion.grad.clone()
        self._omega_grd += self._omega.grad.clone()

        for name, W in self.rgb_decoder.named_parameters():
            if "weight" in name:
                self.rgb_grd[name] = self.rgb_grd[name] + W.grad.clone()

    def zero_gradient_cache(self):

        self._xyz_grd = torch.zeros_like(self._xyz, requires_grad=False)
        self._features_dc_grd = torch.zeros_like(self._features_dc, requires_grad=False)
        self._features_t_grd = torch.zeros_like(self._features_t, requires_grad=False)

        self._scaling_grd = torch.zeros_like(self._scaling, requires_grad=False)
        self._rotation_grd = torch.zeros_like(self._rotation, requires_grad=False)
        self._opacity_grd = torch.zeros_like(self._opacity, requires_grad=False)
        self._trbf_center_grd = torch.zeros_like(self._trbf_center, requires_grad=False)
        self._trbf_scale_grd = torch.zeros_like(self._trbf_scale, requires_grad=False)
        self._motion_grd = torch.zeros_like(self._motion, requires_grad=False)
        self._omega_grd = torch.zeros_like(self._omega, requires_grad=False)

        for name in self.rgb_grd.keys():
            self.rgb_grd[name].zero_()

    def set_batch_gradient(self, cnt):
        ratio = 1 / cnt
        self._features_dc.grad = self._features_dc_grd * ratio
        self._features_t.grad = self._features_t_grd * ratio
        self._xyz.grad = self._xyz_grd * ratio
        self._scaling.grad = self._scaling_grd * ratio
        self._rotation.grad = self._rotation_grd * ratio
        self._opacity.grad = self._opacity_grd * ratio
        self._trbf_center.grad = self._trbf_center_grd * ratio
        self._trbf_scale.grad = self._trbf_scale_grd * ratio
        self._motion.grad = self._motion_grd * ratio
        self._omega.grad = self._omega_grd * ratio

        for name, W in self.rgb_decoder.named_parameters():
            if "weight" in name:
                W.grad = self.rgb_grd[name] * ratio

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.rgb_decoder.cuda()
        # self._features_t
        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_t], "lr": training_args.feature_t_lr, "name": "f_t"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            {"params": [self._omega], "lr": training_args.omega_lr, "name": "omega"},
            {"params": [self._trbf_center], "lr": training_args.trbf_c_lr, "name": "trbf_center"},
            {"params": [self._trbf_scale], "lr": training_args.trbf_s_lr, "name": "trbf_scale"},
            {
                "params": [self._motion],
                "lr": training_args.position_lr_init * self.spatial_lr_scale * 0.5 * training_args.move_lr,
                "name": "motion",
            },
            {"params": list(self.rgb_decoder.parameters()), "lr": training_args.rgb_lr, "name": "decoder"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        print("move decoder to cuda")

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "trbf_center", "trbf_scale", "nx", "ny", "nz"]  # 'trbf_center', 'trbf_scale'
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #     l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        for i in range(self._motion.shape[1]):
            l.append("motion_{}".format(i))

        for i in range(self._features_dc.shape[1]):
            l.append("f_dc_{}".format(i))
        # for i in range(self._features_rest.shape[1]):
        #     l.append('f_rest_{}'.format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        for i in range(self._omega.shape[1]):
            l.append("omega_{}".format(i))

        for i in range(self._features_t.shape[1]):
            l.append("f_t_{}".format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().cpu().numpy()
        # f_rest = self._features_rest.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        trbf_center = self._trbf_center.detach().cpu().numpy()

        trbf_scale = self._trbf_scale.detach().cpu().numpy()
        motion = self._motion.detach().cpu().numpy()

        omega = self._omega.detach().cpu().numpy()

        f_t = self._features_t.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, trbf_center, trbf_scale, normals, motion, f_dc, opacities, scale, rotation, omega, f_t), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

        model_fname = path.replace(".ply", ".pt")
        print(f"Saving model checkpoint to: {model_fname}")
        torch.save(self.rgb_decoder.state_dict(), model_fname)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def zero_omega(self, threshold=0.15):
        scales = self.get_scaling
        omega_mask = torch.sum(torch.abs(self._omega), dim=1) > threshold  # default
        scale_mask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scale_mask_b = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        point_opacity = self.get_opacity
        opacity_mask = point_opacity > 0.7

        mask = torch.logical_and(
            torch.logical_and(omega_mask.unsqueeze(1), scale_mask), torch.logical_and(scale_mask_b, opacity_mask)
        )
        omega_new = mask.float() * self._omega
        optimizable_tensors = self.replace_tensor_to_optimizer(omega_new, "omega")
        self._omega = optimizable_tensors["omega"]
        return mask

    def zero_omega_by_motion(self, threshold=0.15):
        scales = self.get_scaling
        omega_mask = (
            torch.sum(torch.abs(self._motion[:, 0:3]), dim=1) > 0.3
        )  #  #torch.sum(torch.abs(self._omega), dim=1) > threshold # default
        scale_mask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scale_mask_b = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        point_opacity = self.get_opacity
        opacity_mask = point_opacity > 0.7

        mask = torch.logical_and(
            torch.logical_and(omega_mask.unsqueeze(1), scale_mask), torch.logical_and(scale_mask_b, opacity_mask)
        )

        omega_new = mask.float() * self._omega
        optimizable_tensors = self.replace_tensor_to_optimizer(omega_new, "omega")
        self._omega = optimizable_tensors["omega"]
        return mask

    def zero_omega_v2(self, threshold=0.15):
        scales = self.get_scaling
        omega_mask = torch.sum(torch.abs(self._omega), dim=1) > threshold  # default
        scale_mask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scale_mask_b = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        point_opacity = self.get_opacity
        opacity_mask = point_opacity > 0.7

        mask = torch.logical_and(
            torch.logical_and(omega_mask.unsqueeze(1), scale_mask), torch.logical_and(scale_mask_b, opacity_mask)
        )
        omega_new = mask.float() * self._omega
        rotation_new = self.get_rotation(self.delta_t)

        optimizable_tensors = self.replace_tensor_to_optimizer(omega_new, "omega")
        self._omega = optimizable_tensors["omega"]

        optimizable_tensors = self.replace_tensor_to_optimizer(rotation_new, "rotation")
        self._rotation = optimizable_tensors["rotation"]
        return mask

    def load_ply_and_min_max(self, path, max_x, max_y, max_z, min_x, min_y, min_z):
        def logical_or_list(tensor_list):
            mask = None
            for idx, ele in enumerate(tensor_list):
                if idx == 0:
                    mask = ele
                else:
                    mask = np.logical_or(mask, ele)
            return mask

        ply_data = PlyData.read(path)
        ckpt = torch.load(path.replace(".ply", ".pt"))
        self.rgb_decoder.load_state_dict(ckpt)

        xyz = np.stack(
            (
                np.asarray(ply_data.elements[0]["x"]),
                np.asarray(ply_data.elements[0]["y"]),
                np.asarray(ply_data.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(ply_data.elements[0]["opacity"])[..., np.newaxis]
        trbf_center = np.asarray(ply_data.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(ply_data.elements[0]["trbf_scale"])[..., np.newaxis]

        motion_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("motion")]
        num_motion = 9
        motion = np.zeros((xyz.shape[0], num_motion))
        for i in range(num_motion):
            motion[:, i] = np.asarray(ply_data.elements[0]["motion_" + str(i)])

        dc_f_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(ply_data.elements[0]["f_dc_" + str(i)])

        extra_f_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_rest_")]
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(ply_data.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        rot_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        omega_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        ft_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_t")]
        ft_omegas = np.zeros((xyz.shape[0], len(ft_names)))
        for idx, attr_name in enumerate(ft_names):
            ft_omegas[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        mask0 = xyz[:, 0] > max_x.item()
        mask1 = xyz[:, 1] > max_y.item()
        mask2 = xyz[:, 2] > max_z.item()

        mask3 = xyz[:, 0] < min_x.item()
        mask4 = xyz[:, 1] < min_y.item()
        mask5 = xyz[:, 2] < min_z.item()
        mask = logical_or_list([mask0, mask1, mask2, mask3, mask4, mask5])
        mask = np.logical_not(mask)

        self._xyz = nn.Parameter(torch.tensor(xyz[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc[mask], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities[mask], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(torch.tensor(scales[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_center = nn.Parameter(
            torch.tensor(trbf_center[mask], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._trbf_scale = nn.Parameter(
            torch.tensor(trbf_scale[mask], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._motion = nn.Parameter(torch.tensor(motion[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_t = nn.Parameter(
            torch.tensor(ft_omegas[mask], dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def load_ply_and_min_max_Y(self, path, max_x, max_y, max_z, min_x, min_y, min_z):
        def logical_or_list(tensor_list):
            mask = None
            for idx, ele in enumerate(tensor_list):
                if idx == 0:
                    mask = ele
                else:
                    mask = np.logical_or(mask, ele)
            return mask

        ply_data = PlyData.read(path)
        ckpt = torch.load(path.replace(".ply", ".pt"))
        self.rgb_decoder.load_state_dict(ckpt)

        xyz = np.stack(
            (
                np.asarray(ply_data.elements[0]["x"]),
                np.asarray(ply_data.elements[0]["y"]),
                np.asarray(ply_data.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(ply_data.elements[0]["opacity"])[..., np.newaxis]
        trbf_center = np.asarray(ply_data.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(ply_data.elements[0]["trbf_scale"])[..., np.newaxis]

        motion_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("motion")]
        num_motion = 9
        motion = np.zeros((xyz.shape[0], num_motion))
        for i in range(num_motion):
            motion[:, i] = np.asarray(ply_data.elements[0]["motion_" + str(i)])

        dc_f_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(ply_data.elements[0]["f_dc_" + str(i)])

        extra_f_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_rest_")]
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(ply_data.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        rot_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        omega_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        ft_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_t")]
        ft_omegas = np.zeros((xyz.shape[0], len(ft_names)))
        for idx, attr_name in enumerate(ft_names):
            ft_omegas[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        mask1 = xyz[:, 1] > max_y.item()

        mask4 = xyz[:, 1] < min_y.item()
        mask = logical_or_list([mask1, mask4])
        mask = np.logical_not(mask)

        self._xyz = nn.Parameter(torch.tensor(xyz[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc[mask], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(
            torch.tensor(opacities[mask], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(torch.tensor(scales[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_center = nn.Parameter(
            torch.tensor(trbf_center[mask], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._trbf_scale = nn.Parameter(
            torch.tensor(trbf_scale[mask], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._motion = nn.Parameter(torch.tensor(motion[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_t = nn.Parameter(
            torch.tensor(ft_omegas[mask], dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def load_ply_and_min_max_all(self, path, max_x, max_y, max_z, min_x, min_y, min_z):
        def logical_or_list(tensor_list):
            mask = None
            for idx, ele in enumerate(tensor_list):
                if idx == 0:
                    mask = ele
                else:
                    mask = np.logical_or(mask, ele)
            return mask

        ply_data = PlyData.read(path)
        ckpt = torch.load(path.replace(".ply", ".pt"))
        self.rgb_decoder.load_state_dict(ckpt)

        xyz = np.stack(
            (
                np.asarray(ply_data.elements[0]["x"]),
                np.asarray(ply_data.elements[0]["y"]),
                np.asarray(ply_data.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(ply_data.elements[0]["opacity"])[..., np.newaxis]
        trbf_center = np.asarray(ply_data.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(ply_data.elements[0]["trbf_scale"])[..., np.newaxis]

        motion_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("motion")]
        num_motion = 9
        motion = np.zeros((xyz.shape[0], num_motion))
        for i in range(num_motion):
            motion[:, i] = np.asarray(ply_data.elements[0]["motion_" + str(i)])

        dc_f_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(ply_data.elements[0]["f_dc_" + str(i)])

        extra_f_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_rest_")]
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(ply_data.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        rot_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        omega_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        ft_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_t")]
        ft_omegas = np.zeros((xyz.shape[0], len(ft_names)))
        for idx, attr_name in enumerate(ft_names):
            ft_omegas[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        mask0 = xyz[:, 0] > max_x.item()
        mask1 = xyz[:, 1] > max_y.item()
        mask2 = xyz[:, 2] > max_z.item()

        mask3 = xyz[:, 0] < min_x.item()
        mask4 = xyz[:, 1] < min_y.item()
        mask5 = xyz[:, 2] < min_z.item()
        mask = logical_or_list([mask0, mask1, mask2, mask3, mask4, mask5])
        # mask = np.logical_not(mask)# now the reset point is within the boundary

        unstable_points = np.sum(np.abs(motion[:, 0:3]), axis=1)
        moving_points = unstable_points > 0.03
        trbf_mask = trbf_scale < 3  # temporal unstable points

        mask_st = np.logical_or(trbf_mask.squeeze(1), moving_points)

        mask = np.logical_or(mask, mask_st)  # only use large t_scale points.
        # replace points with input ?

        mask = np.logical_not(mask)  # remaining good points. todo remove good mask's NN

        xyz = torch.cat((self._xyz, torch.tensor(xyz[mask], dtype=torch.float, device="cuda")))

        self._xyz = nn.Parameter(xyz.requires_grad_(True))

        features_dc = torch.cat((self._features_dc, torch.tensor(features_dc[mask], dtype=torch.float, device="cuda")))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))

        opacities = torch.cat((self._opacity, torch.tensor(opacities[mask], dtype=torch.float, device="cuda")))
        self._opacity = nn.Parameter(opacities).requires_grad_(True)

        scales = torch.cat((self._scaling, torch.tensor(scales[mask], dtype=torch.float, device="cuda")))

        self._scaling = nn.Parameter(scales).requires_grad_(True)
        rots = torch.cat((self._rotation, torch.tensor(rots[mask], dtype=torch.float, device="cuda")))

        self._rotation = nn.Parameter(rots).requires_grad_(True)
        trbf_center = torch.cat((self._trbf_center, torch.tensor(trbf_center[mask], dtype=torch.float, device="cuda")))
        self._trbf_center = nn.Parameter(trbf_center).requires_grad_(True)
        trbf_scale = torch.cat((self._trbf_scale, torch.tensor(trbf_scale[mask], dtype=torch.float, device="cuda")))

        self._trbf_scale = nn.Parameter(trbf_scale.requires_grad_(True))

        motion = torch.cat((self._motion, torch.tensor(motion[mask], dtype=torch.float, device="cuda")))

        self._motion = nn.Parameter(motion.requires_grad_(True))
        omegas = torch.cat((self._omega, torch.tensor(omegas[mask], dtype=torch.float, device="cuda")))
        self._omega = nn.Parameter(omegas.requires_grad_(True))

        ft_omegas = torch.cat((self._features_t, torch.tensor(ft_omegas[mask], dtype=torch.float, device="cuda")))
        self._features_t = nn.Parameter(ft_omegas.requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def load_ply(self, path):
        ply_data = PlyData.read(path)
        ckpt = torch.load(path.replace(".ply", ".pt"))
        self.rgb_decoder.load_state_dict(ckpt)

        xyz = np.stack(
            (
                np.asarray(ply_data.elements[0]["x"]),
                np.asarray(ply_data.elements[0]["y"]),
                np.asarray(ply_data.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(ply_data.elements[0]["opacity"])[..., np.newaxis]
        trbf_center = np.asarray(ply_data.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(ply_data.elements[0]["trbf_scale"])[..., np.newaxis]

        motion_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("motion")]
        num_motion = 9
        motion = np.zeros((xyz.shape[0], num_motion))
        for i in range(num_motion):
            motion[:, i] = np.asarray(ply_data.elements[0]["motion_" + str(i)])

        dc_f_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(ply_data.elements[0]["f_dc_" + str(i)])

        extra_f_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_rest_")]
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(ply_data.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        rot_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        omega_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        ft_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_t")]
        ft_omegas = np.zeros((xyz.shape[0], len(ft_names)))
        for idx, attr_name in enumerate(ft_names):
            ft_omegas[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_center = nn.Parameter(
            torch.tensor(trbf_center, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._trbf_scale = nn.Parameter(
            torch.tensor(trbf_scale, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._motion = nn.Parameter(torch.tensor(motion, dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_t = nn.Parameter(torch.tensor(ft_omegas, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        self.computed_opacity = self.opacity_activation(self._opacity)
        self.computed_scales = torch.exp(self._scaling)  # change not very large
        self.computed_trbf_scale = torch.exp(self._trbf_scale)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] != "decoder":
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        self._features_t = optimizable_tensors["f_t"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.omega_mask is not None:
            self.omega_mask = self.omega_mask[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                    )

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_opacities,
        new_scaling,
        new_rotation,
        new_trbf_center,
        new_trbf_scale,
        new_motion,
        new_omega,
        new_feature_t,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "trbf_center": new_trbf_center,
            "trbf_scale": new_trbf_scale,
            "motion": new_motion,
            "omega": new_omega,
            "f_t": new_feature_t,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_t = optimizable_tensors["f_t"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split_v2(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)  # n,1,1 to n1
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N, 1)
        new_trbf_center = torch.rand_like(new_trbf_center)  # * 0.5
        new_trbf_scale = self._trbf_scale[selected_pts_mask].repeat(N, 1)
        new_motion = self._motion[selected_pts_mask].repeat(N, 1)
        new_omega = self._omega[selected_pts_mask].repeat(N, 1)
        new_feature_t = self._features_t[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_opacity,
            new_scaling,
            new_rotation,
            new_trbf_center,
            new_trbf_scale,
            new_motion,
            new_omega,
            new_feature_t,
        )

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool))
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # new_trbf_center =  torch.rand((self._trbf_center[selected_pts_mask].shape[0], 1), device="cuda")  #self._trbf_center[selected_pts_mask]
        # new_trbf_center =  self._trbf_center[selected_pts_mask] #
        new_trbf_center = torch.rand(
            (self._trbf_center[selected_pts_mask].shape[0], 1), device="cuda"
        )  # self._trbf_center[selected_pts_mask]
        new_trbf_scale = self._trbf_scale[selected_pts_mask]
        new_motion = self._motion[selected_pts_mask]
        new_omega = self._omega[selected_pts_mask]
        new_feature_t = self._features_t[selected_pts_mask]
        N, c = new_feature_t.shape
        # self.trbf_output = torch.cat((self.trbf_output, torch.zeros(N , 1).to(self.trbf_output)))
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_opacities,
            new_scaling,
            new_rotation,
            new_trbf_center,
            new_trbf_scale,
            new_motion,
            new_omega,
            new_feature_t,
        )

    def densify_prune_clone(self, max_grad, min_opacity, extent, max_screen_size, splitN=1):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        print("before clone", self._xyz.shape[0])
        self.densify_and_clone(grads, max_grad, extent)
        print("after clone", self._xyz.shape[0])

        self.densify_and_split_v2(grads, max_grad, extent, 2)
        print("after split", self._xyz.shape[0])

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def add_gaussians(
        self,
        bad_uv_idx,
        viewpoint_cam,
        depth_map,
        gt_image,
        new_ray_step=3,
        ray_end=2,
        trbf_center=0.5,
        depth_max=None,
        shuffle=False,
    ):
        def pix2ndc(v, S):
            return (v * 2.0 + 1.0) / S - 1.0

        ray_list = torch.linspace(self.ray_start, ray_end, new_ray_step)  # 0.7 to ray_end
        rgbs = gt_image[:, bad_uv_idx[:, 0], bad_uv_idx[:, 1]]
        rgbs = rgbs.permute(1, 0)
        # should we add the feature dc with non zero values?
        feature_dc = torch.cat((rgbs, torch.zeros_like(rgbs)), dim=1)

        depths = depth_map[:, bad_uv_idx[:, 0], bad_uv_idx[:, 1]]
        depths = depths.permute(1, 0)  # only use depth map > 15 .

        depths = torch.ones_like(depths) * depth_max  # use the max local depth for the scene ?

        u = bad_uv_idx[:, 0]  # hight y
        v = bad_uv_idx[:, 1]  # width  x
        N_points = u.shape[0]

        new_xyz = []
        new_scaling = []
        new_rotation = []
        new_features_dc = []
        new_opacity = []
        new_trbf_center = []
        new_trbf_scale = []
        new_motion = []
        new_omega = []
        new_feature_t = []

        camera2wold = viewpoint_cam.world_view_transform.T.inverse()
        project_inverse = viewpoint_cam.projection_matrix.T.inverse()
        max_z, min_z = self.max_z, self.min_z
        max_y, min_y = self.max_y, self.min_y
        max_x, min_x = self.max_x, self.min_x

        for z_scale in ray_list:
            ndc_u, ndc_v = pix2ndc(u, viewpoint_cam.image_height), pix2ndc(v, viewpoint_cam.image_width)
            # targetPz = depths*z_scale # depth in local cameras..
            if shuffle == True:
                random_depth = torch.rand_like(depths) - 0.5  # -0.5 to 0.5
                targetPz = (depths + depths / 10 * (random_depth)) * z_scale
            else:
                targetPz = depths * z_scale  # depth in local cameras..

            ndc_u = ndc_u.unsqueeze(1)
            ndc_v = ndc_v.unsqueeze(1)

            # N,4 ...
            ndc_camera = torch.cat((ndc_v, ndc_u, torch.ones_like(ndc_u) * (1.0), torch.ones_like(ndc_u)), 1)

            local_point_uv = ndc_camera @ project_inverse.T

            direction_in_local = local_point_uv / local_point_uv[:, 3:]  # ray direction in camera space

            rate = targetPz / direction_in_local[:, 2:3]  #

            local_point = direction_in_local * rate

            local_point[:, -1] = 1

            world_point_H = local_point @ camera2wold.T
            world_point = world_point_H / world_point_H[:, 3:]  #

            xyz = world_point[:, :3]
            distance_to_camera_center = viewpoint_cam.camera_center - xyz
            distance_to_camera_center = torch.norm(distance_to_camera_center, dim=1)

            x_mask = torch.logical_and(xyz[:, 0] > min_x, xyz[:, 0] < max_x)
            selected_mask = torch.logical_or(x_mask, torch.logical_not(x_mask))  # torch.logical_and(x_mask, y_mask)
            new_xyz.append(xyz[selected_mask])

            new_features_dc.append(feature_dc.cuda(0)[selected_mask])

            select_num_points = torch.sum(selected_mask).item()
            new_trbf_center.append(torch.rand((select_num_points, 1)).cuda())

            assert self.trbf_scale_init < 1
            new_trbf_scale.append(self.trbf_scale_init * torch.ones((select_num_points, 1), device="cuda"))
            new_motion.append(torch.zeros((select_num_points, 9), device="cuda"))
            new_omega.append(torch.zeros((select_num_points, 4), device="cuda"))
            new_feature_t.append(torch.zeros((select_num_points, 3), device="cuda"))

        new_xyz = torch.cat(new_xyz, dim=0)
        new_rotation = torch.zeros((new_xyz.shape[0], 4), device="cuda")
        new_rotation[:, 1] = 0

        new_features_dc = torch.cat(new_features_dc, dim=0)
        new_opacity = inverse_sigmoid(0.1 * torch.ones_like(new_xyz[:, 0:1]))
        new_trbf_center = torch.cat(new_trbf_center, dim=0)
        new_trbf_scale = torch.cat(new_trbf_scale, dim=0)
        new_motion = torch.cat(new_motion, dim=0)
        new_omega = torch.cat(new_omega, dim=0)
        new_feature_t = torch.cat(new_feature_t, dim=0)

        tmp_xyz = torch.cat((new_xyz, self._xyz), dim=0)
        dist2 = torch.clamp_min(distCUDA2(tmp_xyz), 0.0000001)
        dist2 = dist2[: new_xyz.shape[0]]
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)
        new_scaling = scales

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_opacity,
            new_scaling,
            new_rotation,
            new_trbf_center,
            new_trbf_scale,
            new_motion,
            new_omega,
            new_feature_t,
        )
        return new_xyz.shape[0]

    def prune_points_with_ems_mask(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        self._features_t = optimizable_tensors["f_t"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self.mask_for_ems = self.mask_for_ems[valid_points_mask]  # we only remain valid mask from ems
