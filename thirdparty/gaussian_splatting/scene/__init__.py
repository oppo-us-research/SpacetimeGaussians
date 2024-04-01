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

import json
import os
import random

import torch

from PIL import Image

from helper_train import get_fish_eye_mapper, record_points_helper
from thirdparty.gaussian_splatting.arguments import ModelParams
from thirdparty.gaussian_splatting.scene.dataset_readers import (
    scene_load_type_callbacks,
)
from thirdparty.gaussian_splatting.scene.oursfull import GaussianModel
from thirdparty.gaussian_splatting.utils.camera_utils import (
    camera_list_from_cam_infos,
    camera_list_from_cam_infos_v2,
    camera_list_from_cam_infos_v2_no_gt,
    camera_to_json,
)
from thirdparty.gaussian_splatting.utils.system_utils import search_for_max_iteration


class Scene:

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        multi_view=False,
        duration=50,
        loader="colmap",
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.ref_model_path = None

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = search_for_max_iteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        ray_dict = {}

        if loader == "colmap" or loader == "colmap_valid":  # colmap_valid only for testing
            scene_info = scene_load_type_callbacks["colmap"](
                args.source_path, args.images, args.eval, multi_view, duration=duration
            )

        elif loader == "technicolor" or loader == "technicolor_valid":
            scene_info = scene_load_type_callbacks["technicolor"](
                args.source_path, args.images, args.eval, multi_view, duration=duration
            )

        elif loader == "immersive" or loader == "immersive_valid" or loader == "immersive_ss":
            scene_info = scene_load_type_callbacks["immersive"](
                args.source_path, args.images, args.eval, multi_view, duration=duration
            )
        elif loader == "immersive_valid_ss":
            scene_info = scene_load_type_callbacks["immersive"](
                args.source_path, args.images, args.eval, multi_view, duration=duration, test_only=True
            )

        elif loader == "colmapmv":  # colmap_valid only for testing
            scene_info = scene_load_type_callbacks["colmapmv"](
                args.source_path, args.images, args.eval, multi_view, duration=duration
            )

        elif loader == "hyfluid" or loader == "hyfluid_valid":
            scene_info = scene_load_type_callbacks["hyfluid"](
                args.source_path, args.white_background, args.eval, duration=duration
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            cam_list = []
            if scene_info.test_cameras:
                cam_list.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                cam_list.extend(scene_info.train_cameras)
            for id, cam in enumerate(cam_list):
                json_cams.append(camera_to_json(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file, indent=2)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            if loader in [
                "colmap_valid",
                "colmapmv",
                "immersive_valid",
                "technicolor_valid",
                "immersive_valid_ss",
                "imv2valid",
                "hyfluid_valid",
            ]:
                self.train_cameras[resolution_scale] = []  # no training data

            elif loader in ["immersive_ss"]:
                assert resolution_scale == 1.0, "High frequency data only available at 1.0 scale"
                self.train_cameras[resolution_scale] = camera_list_from_cam_infos_v2(
                    scene_info.train_cameras, resolution_scale, args, ss=True
                )

            else:
                self.train_cameras[resolution_scale] = camera_list_from_cam_infos_v2(
                    scene_info.train_cameras, resolution_scale, args
                )

            print("Loading Test Cameras")
            if loader in [
                "colmap_valid",
                "immersive_valid",
                "colmap",
                "technicolor_valid",
                "technicolor",
                "imv2",
                "imv2valid",
                "hyfluid",
                "hyfluid_valid",
            ]:
                # we need gt for metrics
                self.test_cameras[resolution_scale] = camera_list_from_cam_infos_v2(
                    scene_info.test_cameras, resolution_scale, args
                )
            elif loader in ["immersive_ss", "immersive_valid_ss"]:
                self.test_cameras[resolution_scale] = camera_list_from_cam_infos_v2(
                    scene_info.test_cameras, resolution_scale, args, ss=True
                )
            elif loader in ["colmapmv"]:  # only for multi view
                self.test_cameras[resolution_scale] = camera_list_from_cam_infos_v2_no_gt(
                    scene_info.test_cameras, resolution_scale, args
                )

        for cam in self.train_cameras[resolution_scale]:
            if cam.image_name not in ray_dict and cam.rayo is not None:
                # rays_o, rays_d = 1, camera_direct
                ray_dict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda()  # 1 x 6 x H x W

        for cam in self.test_cameras[resolution_scale]:
            if cam.image_name not in ray_dict and cam.rayo is not None:
                ray_dict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda()  # 1 x 6 x H x W

        for cam in self.train_cameras[resolution_scale]:
            cam.rays = ray_dict[cam.image_name]  # should be direct ?

        for cam in self.test_cameras[resolution_scale]:
            cam.rays = ray_dict[cam.image_name]  # should be direct ?

        if loader in ["immersive_ss", "immersive_valid_ss"]:  # construct shared fish_eyed remapping
            self.fish_eye_mapper = {}
            for cam in self.train_cameras[resolution_scale]:
                if cam.image_name not in self.fish_eye_mapper:
                    self.fish_eye_mapper[cam.image_name] = get_fish_eye_mapper(args.source_path, cam.image_name)  #
                    self.fish_eye_mapper[cam.image_name].requires_grad = False

            for cam in self.test_cameras[resolution_scale]:
                if cam.image_name not in self.fish_eye_mapper:
                    self.fish_eye_mapper[cam.image_name] = get_fish_eye_mapper(args.source_path, cam.image_name)  #
                    self.fish_eye_mapper[cam.image_name].requires_grad = False

            for cam in self.train_cameras[resolution_scale]:
                cam.fish_eye_mapper = self.fish_eye_mapper[cam.image_name]
            for cam in self.test_cameras[resolution_scale]:
                cam.fish_eye_mapper = self.fish_eye_mapper[cam.image_name]

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    # record_points_helper(model_path, numpoints, iteration, string):
    def record_points(self, iteration, string):
        txt_path = os.path.join(self.model_path, "exp_log.txt")
        numpoints = self.gaussians._xyz.shape[0]
        record_points_helper(self.model_path, numpoints, iteration, string)

    def get_train_cameras(self, scale=1.0):
        return self.train_cameras[scale]

    def get_test_cameras(self, scale=1.0):
        return self.test_cameras[scale]
