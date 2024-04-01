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

import glob
import json
import os
import sys

from pathlib import Path
from typing import NamedTuple

import natsort
import numpy as np
import torch

from PIL import Image
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

from thirdparty.gaussian_splatting.scene.colmap_loader import (
    qvec_2_rot_mat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from thirdparty.gaussian_splatting.utils.graphics_utils import (
    BasicPointCloud,
    focal2fov,
    fov2focal,
    get_world_2_view2,
)
from thirdparty.gaussian_splatting.utils.sh_utils import sh2rgb


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    near: float
    far: float
    timestamp: float
    pose: np.array
    hp_directions: np.array
    cxr: float
    cyr: float


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def get_nerf_pp_norm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = get_world_2_view2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def read_colmap_cameras(cam_extrinsics, cam_intrinsics, images_folder, near, far, start_time=0, duration=50):
    cam_infos = []

    # pose in llff. pipeline by hypereel: https://github.com/facebookresearch/hyperreel
    origin_numpy = os.path.join(os.path.dirname(os.path.dirname(images_folder)), "poses_bounds.npy")
    with open(origin_numpy, "rb") as numpy_file:
        poses_bounds = np.load(numpy_file)

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        bounds = poses_bounds[:, -2:]

        near = bounds.min() * 0.95
        far = bounds.max() * 1.05

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # 19, 3, 5

        H, W, focal = poses[0, :, -1]
        cx, cy = W / 2.0, H / 2.0

        K = np.eye(3)
        K[0, 0] = focal * W / W / 2.0
        K[0, 2] = cx * W / W / 2.0
        K[1, 1] = focal * H / H / 2.0
        K[1, 2] = cy * H / H / 2.0

        imageH = int(H // 2)  # note hard coded to half of the original image size
        imageW = int(W // 2)

    total_cam_name = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        total_cam_name.append(extr.name)

    sorted_total_cam_name_list = natsort.natsorted(total_cam_name)
    sorted_name_dict = {}
    for i in range(len(sorted_total_cam_name_list)):
        sorted_name_dict[sorted_total_cam_name_list[i]] = i  # map each cam with a number

    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec_2_rot_mat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        for j in range(start_time, start_time + int(duration)):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image_path = image_path.replace("colmap_" + str(start_time), "colmap_{}".format(j), 1)
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            image = Image.open(image_path)
            if j == start_time:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    near=near,
                    far=far,
                    timestamp=(j - start_time) / duration,
                    pose=1,
                    hp_directions=1,
                    cxr=0.0,
                    cyr=0.0,
                )

            else:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    near=near,
                    far=far,
                    timestamp=(j - start_time) / duration,
                    pose=None,
                    hp_directions=None,
                    cxr=0.0,
                    cyr=0.0,
                )
            cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def read_colmap_cameras_technicolor(
    cam_extrinsics, cam_intrinsics, images_folder, near, far, start_time=0, duration=50
):

    cam_infos = []
    total_cam_name = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        total_cam_name.append(extr.name)

    sorted_total_cam_name_list = natsort.natsorted(total_cam_name)
    sorted_name_dict = {}
    for i in range(len(sorted_total_cam_name_list)):
        sorted_name_dict[sorted_total_cam_name_list[i]] = i  # map each cam with a number

    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec_2_rot_mat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        for j in range(start_time, start_time + int(duration)):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image_path = image_path.replace("colmap_" + str(start_time), "colmap_{}".format(j), 1)

            cxr = (intr.params[2]) / width - 0.5
            cyr = (intr.params[3]) / height - 0.5

            K = np.eye(3)
            K[0, 0] = focal_length_x  # * 0.5
            K[0, 2] = intr.params[2]  # * 0.5
            K[1, 1] = focal_length_y  # * 0.5
            K[1, 2] = intr.params[3]  # * 0.5

            halfH = round(height / 2.0)
            halfW = round(width / 2.0)

            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)

            image = Image.open(image_path)

            if j == start_time:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    near=near,
                    far=far,
                    timestamp=(j - start_time) / duration,
                    pose=1,
                    hp_directions=1,
                    cxr=cxr,
                    cyr=cyr,
                )
            else:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    near=near,
                    far=far,
                    timestamp=(j - start_time) / duration,
                    pose=None,
                    hp_directions=None,
                    cxr=cxr,
                    cyr=cyr,
                )
            cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def normalize(v):
    return v / np.linalg.norm(v)


def read_colmap_cameras_mv(cam_extrinsics, cam_intrinsics, images_folder, near, far, start_time=0, duration=50):
    cam_infos = []
    from utils.graphics_utils import (
        get_projection_matrix,
        get_projection_matrix_cv,
        get_world_2_view2,
    )

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec_2_rot_mat(extr.qvec))
        T = np.array(extr.tvec)

        world_view_transform = torch.tensor(get_world_2_view2(R, T)).transpose(0, 1).cuda()

        cxr = (intr.params[2]) / width - 0.5
        cyr = (intr.params[3]) / height - 0.5

        if extr.name == "cam00.png":
            if intr.model == "SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert (
                    False
                ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            if cyr != 0.0:
                cxr = cxr
                cyr = cyr
                projection_matrix = (
                    get_projection_matrix_cv(z_near=0.01, z_far=100.0, fovX=FovX, fovY=FovY, cx=cxr, cy=cyr)
                    .transpose(0, 1)
                    .cuda()
                )
            else:
                projection_matrix = (
                    get_projection_matrix(z_near=0.01, z_far=100.0, fovX=FovX, fovY=FovY).transpose(0, 1).cuda()
                )

            camera_center = world_view_transform.inverse()[3, :3]

            project_inverse = projection_matrix.T.inverse()
            camera2wold = world_view_transform.T.inverse()

            ndc_camera = torch.Tensor((0, 0, 1, 1)).cuda()
            ndc_camera = ndc_camera.unsqueeze(0)  # 1, 4
            projected = ndc_camera @ project_inverse.T  # 1, 4,  @ 4,4
            direction_in_local = projected / projected[:, 3:]  # v
            direction = direction_in_local[:, :3] @ camera2wold[:3, :3].T
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

            target = camera_center + rays_d * 30.0
            break

    radiance = 1.0

    for i in range(240):
        theta = i / 240.0 * 4 * np.pi
        new_camera_center = (
            camera_center + radiance * torch.Tensor((np.cos(theta), np.sin(theta), 2.0 + 2.0 * np.sin(theta))).cuda()
        )

        new_forward_vector = target - new_camera_center
        new_forward_vector = new_forward_vector.cpu().numpy()

        right_vector = R[:, 0]  # First column
        up_vector = R[:, 1]  # Second column
        forward_vector = R[:, 2]  # Third column

        new_right = normalize(np.cross(up_vector, new_forward_vector))
        up = normalize(np.cross(new_forward_vector, new_right))

        newR = np.eye(3)
        newR[:, 0] = new_right
        newR[:, 1] = up
        newR[:, 2] = normalize(new_forward_vector)

        C2W = np.zeros((4, 4))
        C2W[:3, :3] = newR
        C2W[:3, 3] = new_camera_center.cpu().numpy()
        C2W[3, 3] = 1.0
        rt = np.linalg.inv(C2W)
        newt = rt[:3, 3]

        image_name = "mv_" + str(i)
        uid = i

        time = (i) / 240
        cam_info = CameraInfo(
            uid=uid,
            R=newR,
            T=newt,
            FovY=FovY,
            FovX=FovX,
            image=None,
            image_path=None,
            image_name=image_name,
            width=width,
            height=height,
            near=near,
            far=far,
            timestamp=time,
            pose=1,
            hp_directions=0,
            cxr=0.0,
            cyr=0.0,
        )

        cam_infos.append(cam_info)

    sys.stdout.write("\n")
    return cam_infos


def read_colmap_cameras_immersive(cam_extrinsics, cam_intrinsics, images_folder, near, far, start_time=0, duration=50):
    cam_infos = []

    total_cam_name = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        total_cam_name.append(extr.name)

    sorted_total_cam_name_list = natsort.natsorted(total_cam_name)
    sorted_name_dict = {}
    for i in range(len(sorted_total_cam_name_list)):
        sorted_name_dict[sorted_total_cam_name_list[i]] = i  # map each cam with a number

    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec_2_rot_mat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # if extr.name not in ["camera_0005.png", "camera_0001.png"]:
        #         continue

        for j in range(start_time, start_time + int(duration)):
            # image_path = os.path.join(images_folder, os.path.basename(extr.name))
            # image_name = os.path.basename(image_path).split(".")[0]
            # image_path = image_path.replace("colmap_"+str(start_time), "colmap_{}".format(j), 1)

            parent_folder = os.path.dirname(images_folder)
            parent_folder = os.path.dirname(parent_folder)
            image_name = extr.name.split(".")[0]

            raw_video_folder = os.path.join(parent_folder, os.path.basename(image_name))

            image_path = os.path.join(raw_video_folder, str(j) + ".png")

            # image_path.replace("colmap_"+str(start_time), "colmap_{}".format(j), 1)

            # K = np.eye(3)
            # K[0, 0] = focal_length_x * 0.5
            # K[0, 2] = intr.params[2] * 0.5
            # K[1, 1] = focal_length_y * 0.5
            # K[1, 2] = intr.params[3] * 0.5

            cxr = (intr.params[2]) / width - 0.5
            cyr = (intr.params[3]) / height - 0.5

            K = np.eye(3)
            K[0, 0] = focal_length_x  # * 0.5
            K[0, 2] = intr.params[2]  # * 0.5
            K[1, 1] = focal_length_y  # * 0.5
            K[1, 2] = intr.params[3]  # * 0.5

            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            image = Image.open(image_path)
            if j == start_time:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    near=near,
                    far=far,
                    timestamp=(j - start_time) / duration,
                    pose=1,
                    hp_directions=1,
                    cxr=cxr,
                    cyr=cyr,
                )
            else:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    near=near,
                    far=far,
                    timestamp=(j - start_time) / duration,
                    pose=None,
                    hp_directions=None,
                    cxr=cxr,
                    cyr=cyr,
                )
            cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def read_colmap_cameras_immersive_test_only(
    cam_extrinsics, cam_intrinsics, images_folder, near, far, start_time=0, duration=50
):
    cam_infos = []

    total_cam_name = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        total_cam_name.append(extr.name)

    sorted_total_cam_name_list = natsort.natsorted(total_cam_name)
    sorted_name_dict = {}
    for i in range(len(sorted_total_cam_name_list)):
        sorted_name_dict[sorted_total_cam_name_list[i]] = i  # map each cam with a number

    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec_2_rot_mat(extr.qvec))
        T = np.array(extr.tvec)
        # T[1] = T[1] + 0.2  # y increase by 1
        # T[2] = T[2] + 0.65
        # T[0] = T[0] + 0.65 # x by 0.65
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        for j in range(start_time, start_time + int(duration)):
            # image_path = os.path.join(images_folder, os.path.basename(extr.name))
            # image_name = os.path.basename(image_path).split(".")[0]
            # image_path = image_path.replace("colmap_"+str(start_time), "colmap_{}".format(j), 1)

            parent_folder = os.path.dirname(images_folder)
            parent_folder = os.path.dirname(parent_folder)
            image_name = extr.name.split(".")[0]

            raw_video_folder = os.path.join(parent_folder, os.path.basename(image_name))

            image_path = os.path.join(raw_video_folder, str(j) + ".png")

            # image_path.replace("colmap_"+str(start_time), "colmap_{}".format(j), 1)

            # K = np.eye(3)
            # K[0, 0] = focal_length_x * 0.5
            # K[0, 2] = intr.params[2] * 0.5
            # K[1, 1] = focal_length_y * 0.5
            # K[1, 2] = intr.params[3] * 0.5

            cxr = (intr.params[2]) / width - 0.5
            cyr = (intr.params[3]) / height - 0.5

            K = np.eye(3)
            K[0, 0] = focal_length_x  # * 0.5
            K[0, 2] = intr.params[2]  # * 0.5
            K[1, 1] = focal_length_y  # * 0.5
            K[1, 2] = intr.params[3]  # * 0.5

            # halfH = round(height / 2.0 )
            # halfW = round(width / 2.0 )

            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)

            if image_name == "camera_0001":
                image = Image.open(image_path)
            else:
                image = None
            if j == start_time:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    near=near,
                    far=far,
                    timestamp=(j - start_time) / duration,
                    pose=1,
                    hp_directions=1,
                    cxr=cxr,
                    cyr=cyr,
                )
            else:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    near=near,
                    far=far,
                    timestamp=(j - start_time) / duration,
                    pose=None,
                    hp_directions=None,
                    cxr=cxr,
                    cyr=cyr,
                )
            cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetch_ply(path):
    ply_data = PlyData.read(path)
    vertices = ply_data["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    times = np.vstack([vertices["t"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)


def store_ply(path, xyzt, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("t", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    xyz = xyzt[:, :3]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyzt.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzt, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def read_colmap_scene_info_immersive(path, images, eval, llff_hold=8, multi_view=False, duration=50, test_only=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    near = 0.01
    far = 100

    start_time = os.path.basename(path).split("_")[1]  # colmap_0,
    assert start_time.isdigit(), "Colmap folder name must be colmap_<start_time>_<duration>!"
    start_time = int(start_time)

    # read_colmap_cameras_immersive_test_only
    if test_only:
        cam_infos_unsorted = read_colmap_cameras_immersive_test_only(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(path, reading_dir),
            near=near,
            far=far,
            start_time=start_time,
            duration=duration,
        )
    else:
        cam_infos_unsorted = read_colmap_cameras_immersive(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(path, reading_dir),
            near=near,
            far=far,
            start_time=start_time,
            duration=duration,
        )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = cam_infos[duration:]  # + cam_infos[:duration] # for demo only
        test_cam_infos = cam_infos[:duration]
        unique_check = []
        for cam_info in test_cam_infos:
            if cam_info.image_name not in unique_check:
                unique_check.append(cam_info.image_name)
        assert len(unique_check) == 1

        sanity_check = []
        for cam_info in train_cam_infos:
            if cam_info.image_name not in sanity_check:
                sanity_check.append(cam_info.image_name)
        for test_name in unique_check:
            assert test_name not in sanity_check
    else:
        train_cam_infos = cam_infos  # for demo without eval
        test_cam_infos = cam_infos[:duration]

    nerf_normalization = get_nerf_pp_norm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    total_ply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")

    # if os.path.exists(ply_path):
    #     os.remove(ply_path)

    # if os.path.exists(total_ply_path):
    #     os.remove(total_ply_path)

    if not os.path.exists(total_ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        total_xyz = []
        total_rgb = []
        total_time = []

        for i in range(start_time, start_time + duration):
            this_bin_path = os.path.join(path, "sparse/0/points3D.bin").replace(
                "colmap_" + str(start_time), "colmap_" + str(i), 1
            )
            xyz, rgb, _ = read_points3D_binary(this_bin_path)

            total_xyz.append(xyz)
            total_rgb.append(rgb)
            total_time.append(np.ones((xyz.shape[0], 1)) * (i - start_time) / duration)
        xyz = np.concatenate(total_xyz, axis=0)
        rgb = np.concatenate(total_rgb, axis=0)
        total_time = np.concatenate(total_time, axis=0)
        assert xyz.shape[0] == rgb.shape[0]
        xyzt = np.concatenate((xyz, total_time), axis=1)
        store_ply(total_ply_path, xyzt, rgb)
    try:
        pcd = fetch_ply(total_ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=total_ply_path,
    )
    return scene_info


def read_colmap_scene_info_mv(path, images, eval, llff_hold=8, multi_view=False, duration=50):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    near = 0.01
    far = 100

    start_time = os.path.basename(path).split("_")[1]  # colmap_0,
    assert start_time.isdigit(), "Colmap folder name must be colmap_<start_time>_<duration>!"
    start_time = int(start_time)

    cam_infos_unsorted = read_colmap_cameras_mv(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        near=near,
        far=far,
        start_time=start_time,
        duration=duration,
    )
    cam_infos = cam_infos_unsorted

    # for cam in cam_infos:
    #     print(cam.image_name)
    # for cam_info in cam_infos:
    #     print(cam_info.uid, cam_info.R, cam_info.T, cam_info.FovY, cam_info.image_name)

    train_cam_infos = []
    test_cam_infos = cam_infos

    nerf_normalization = get_nerf_pp_norm(test_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    total_ply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + "_mv.ply")

    # if os.path.exists(ply_path):
    #     os.remove(ply_path)

    # if os.path.exists(total_ply_path):
    #     os.remove(total_ply_path)

    # if not os.path.exists(total_ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     total_xyz = []
    #     total_rgb = []
    #     total_time = []
    #     for i in range(start_time, start_time + duration):
    #         this_bin_path = os.path.join(path, "sparse/0/points3D.bin").replace("colmap_"+ str(start_time), "colmap_" + str(i), 1)
    #         xyz, rgb, _ = read_points3D_binary(this_bin_path)
    #         total_xyz.append(xyz)
    #         total_rgb.append(rgb)
    #         total_time.append(np.ones((xyz.shape[0], 1)) * (i-start_time) / duration)
    #     xyz = np.concatenate(total_xyz, axis=0)
    #     rgb = np.concatenate(total_rgb, axis=0)
    #     total_time = np.concatenate(total_time, axis=0)
    #     assert xyz.shape[0] == rgb.shape[0]
    #     xyzt =np.concatenate( (xyz, total_time), axis=1)
    #     store_ply(total_ply_path, xyzt, rgb)
    # try:
    #     pcd = fetch_ply(total_ply_path)
    # except:
    pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=total_ply_path,
    )
    return scene_info


def read_colmap_scene_info(path, images, eval, llff_hold=8, multi_view=False, duration=50):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    near = 0.01
    far = 100

    start_time = os.path.basename(path).split("_")[1]  # colmap_0,
    assert start_time.isdigit(), "Colmap folder name must be colmap_<start_time>_<duration>!"
    start_time = int(start_time)

    cam_infos_unsorted = read_colmap_cameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        near=near,
        far=far,
        start_time=start_time,
        duration=duration,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = cam_infos[duration:]
        test_cam_infos = cam_infos[:duration]
        unique_check = []
        for cam_info in test_cam_infos:
            if cam_info.image_name not in unique_check:
                unique_check.append(cam_info.image_name)
        assert len(unique_check) == 1

        sanity_check = []
        for cam_info in train_cam_infos:
            if cam_info.image_name not in sanity_check:
                sanity_check.append(cam_info.image_name)
        for test_name in unique_check:
            assert test_name not in sanity_check
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos[:2]  # dummy

    nerf_normalization = get_nerf_pp_norm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    total_ply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")

    if not os.path.exists(total_ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        total_xyz = []
        total_rgb = []
        total_time = []
        for i in range(start_time, start_time + duration):
            this_bin_path = os.path.join(path, "sparse/0/points3D.bin").replace(
                "colmap_" + str(start_time), "colmap_" + str(i), 1
            )
            xyz, rgb, _ = read_points3D_binary(this_bin_path)
            print("rgb", rgb.shape, rgb.min(), rgb.max())

            total_xyz.append(xyz)
            total_rgb.append(rgb)
            total_time.append(np.ones((xyz.shape[0], 1)) * (i - start_time) / duration)
        xyz = np.concatenate(total_xyz, axis=0)
        rgb = np.concatenate(total_rgb, axis=0)
        total_time = np.concatenate(total_time, axis=0)
        assert xyz.shape[0] == rgb.shape[0]
        xyzt = np.concatenate((xyz, total_time), axis=1)
        store_ply(total_ply_path, xyzt, rgb)
    try:
        pcd = fetch_ply(total_ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=total_ply_path,
    )
    return scene_info


def read_colmap_scene_info_technicolor(path, images, eval, llff_hold=8, multi_view=False, duration=50):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    start_time = os.path.basename(path).split("_")[1]  # colmap_0,
    assert start_time.isdigit(), "Colmap folder name must be colmap_<start_time>_<duration>!"
    start_time = int(start_time)

    near = 0.01
    far = 100
    cam_infos_unsorted = read_colmap_cameras_technicolor(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        near=near,
        far=far,
        start_time=start_time,
        duration=duration,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    # for cam in cam_infos:
    #     print(cam.image_name)
    # for cam_info in cam_infos:
    #     print(cam_info.uid, cam_info.R, cam_info.T, cam_info.FovY, cam_info.image_name)

    if eval:
        train_cam_infos = [_ for _ in cam_infos if "cam10" not in _.image_name]
        test_cam_infos = [_ for _ in cam_infos if "cam10" in _.image_name]
        unique_check = []
        for cam_info in test_cam_infos:
            if cam_info.image_name not in unique_check:
                unique_check.append(cam_info.image_name)
        assert len(unique_check) == 1

        sanity_check = []
        for cam_info in train_cam_infos:
            if cam_info.image_name not in sanity_check:
                sanity_check.append(cam_info.image_name)
        for test_name in unique_check:
            assert test_name not in sanity_check
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos[:4]

    nerf_normalization = get_nerf_pp_norm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    total_ply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")

    if not os.path.exists(total_ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        total_xyz = []
        total_rgb = []
        total_time = []
        for i in range(start_time, start_time + duration):
            this_bin_path = os.path.join(path, "sparse/0/points3D.bin").replace(
                "colmap_" + str(start_time), "colmap_" + str(i), 1
            )
            xyz, rgb, _ = read_points3D_binary(this_bin_path)
            total_xyz.append(xyz)
            total_rgb.append(rgb)
            total_time.append(np.ones((xyz.shape[0], 1)) * (i - start_time) / duration)
        xyz = np.concatenate(total_xyz, axis=0)
        rgb = np.concatenate(total_rgb, axis=0)
        total_time = np.concatenate(total_time, axis=0)
        assert xyz.shape[0] == rgb.shape[0]
        xyzt = np.concatenate((xyz, total_time), axis=1)
        store_ply(total_ply_path, xyzt, rgb)
    try:
        pcd = fetch_ply(total_ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=total_ply_path,
    )
    return scene_info


def read_cameras_from_transforms(path, transforms_file, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transforms_file)) as json_file:
        contents = json.load(json_file)
        fov_x = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fov_y = focal2fov(fov2focal(fov_x, image.size[0]), image.size[1])
            FovY = fov_y
            FovX = fov_x

            for j in range(20):
                cam_infos.append(
                    CameraInfo(
                        uid=idx * 20 + j,
                        R=R,
                        T=T,
                        FovY=FovY,
                        FovX=FovX,
                        image=image,
                        image_path=image_path,
                        image_name=image_name,
                        width=image.size[0],
                        height=image.size[1],
                    )
                )

    return cam_infos


def read_nerf_synthetic_info(path, white_background, eval, extension=".png", multi_view=False):
    print("Reading Training Transforms")
    train_cam_infos = read_cameras_from_transforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = read_cameras_from_transforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = get_nerf_pp_norm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=sh2rgb(shs), normals=np.zeros((num_pts, 3)))

        store_ply(ply_path, xyz, sh2rgb(shs) * 255)
    try:
        pcd = fetch_ply(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def read_cameras_from_transforms_hyfluid(
    path, transforms_file, white_background, extension=".png", start_time=0, duration=50
):
    cam_infos = []

    with open(os.path.join(path, transforms_file)) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            camera_hw = frame["camera_hw"]
            h, w = camera_hw
            fov_x = frame["camera_angle_x"]
            fov_y = focal2fov(fov2focal(fov_x, w), h)
            FovY = fov_y
            FovX = fov_x
            near = float(frame["near"])
            far = float(frame["far"])

            for time_idx in range(start_time, start_time + duration):

                cam_name = os.path.join(path, f"colmap_{time_idx}", frame["file_path"] + extension)
                timestamp = (time_idx - start_time) / duration
                image_path = os.path.join(path, cam_name)
                image_name = os.path.basename(image_path).split(".")[0]
                image = Image.open(image_path)

                im_data = np.array(image.convert("RGBA"))

                bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

                pose = 1 if time_idx == start_time else None
                hp_directions = 1 if time_idx == start_time else None

                cam_infos.append(
                    CameraInfo(
                        uid=idx * duration + time_idx,
                        R=R,
                        T=T,
                        FovY=FovY,
                        FovX=FovX,
                        image=image,
                        image_path=image_path,
                        image_name=image_name,
                        width=image.size[0],
                        height=image.size[1],
                        timestamp=timestamp,
                        near=near,
                        far=far,
                        pose=pose,
                        hp_directions=hp_directions,
                        cxr=0.0,
                        cyr=0.0,
                    )
                )

    return cam_infos


def read_nerf_synthetic_info_hyfluid(path, white_background, eval, extension=".png", start_time=0, duration=50):
    print("Reading Training Transforms")
    train_cam_infos = read_cameras_from_transforms_hyfluid(
        path, "transforms_train_hyfluid.json", white_background, extension, start_time, duration
    )
    print("Reading Test Transforms")
    test_cam_infos = read_cameras_from_transforms_hyfluid(
        path, "transforms_test_hyfluid.json", white_background, extension, start_time, duration
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = get_nerf_pp_norm(train_cam_infos)

    total_ply_path = os.path.join(path, "points3d_total.ply")
    if os.path.exists(total_ply_path):
        os.remove(total_ply_path)

    if not os.path.exists(total_ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random {duration} point cloud ({num_pts})...")

        # # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        # shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=sh2rgb(shs), normals=np.zeros((num_pts, 3)))

        # store_ply(ply_path, xyz, sh2rgb(shs) * 255)

        total_xyz = []
        total_rgb = []
        total_time = []

        for i in range(start_time, start_time + duration):
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            # rgb = np.random.random((num_pts, 3)) * 255.0
            rgb = sh2rgb(shs) * 255

            total_xyz.append(xyz)
            total_rgb.append(rgb)
            total_time.append(np.ones((xyz.shape[0], 1)) * (i - start_time) / duration)
        xyz = np.concatenate(total_xyz, axis=0)
        rgb = np.concatenate(total_rgb, axis=0)
        total_time = np.concatenate(total_time, axis=0)
        assert xyz.shape[0] == rgb.shape[0]
        xyzt = np.concatenate((xyz, total_time), axis=1)
        store_ply(total_ply_path, xyzt, rgb)
    try:
        pcd = fetch_ply(total_ply_path)
    except:
        pcd = None

    assert pcd is not None, "Point cloud could not be loaded!"

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=total_ply_path,
    )
    return scene_info


def read_colmap_cameras_immersive_v2_test_only(
    cam_extrinsics, cam_intrinsics, images_folder, near, far, start_time=0, duration=50
):
    cam_infos = []

    total_cam_name = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        total_cam_name.append(extr.name)

    sorted_total_cam_name_list = natsort.natsorted(total_cam_name)
    sorted_name_dict = {}
    for i in range(len(sorted_total_cam_name_list)):
        sorted_name_dict[sorted_total_cam_name_list[i]] = i  # map each cam with a number

    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec_2_rot_mat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        for j in range(start_time, start_time + int(duration)):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image_path = image_path.replace("colmap_" + str(start_time), "colmap_{}".format(j), 1)

            # parent_folder = os.path.dirname(images_folder)
            # parent_folder = os.path.dirname(parent_folder)
            # image_name = extr.name.split(".")[0]

            # raw_video_folder = os.path.join(parent_folder,os.path.basename(image_name))

            # image_path = os.path.join(raw_video_folder, str(j) + ".png")

            cxr = (intr.params[2]) / width - 0.5
            cyr = (intr.params[3]) / height - 0.5

            K = np.eye(3)
            K[0, 0] = focal_length_x  # * 0.5
            K[0, 2] = intr.params[2]  # * 0.5
            K[1, 1] = focal_length_y  # * 0.5
            K[1, 2] = intr.params[3]  # * 0.5

            # halfH = round(height / 2.0 )
            # halfW = round(width / 2.0 )

            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)

            if image_name == "camera_0001":
                image = Image.open(image_path)
            else:
                image = None
            if j == start_time:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    near=near,
                    far=far,
                    timestamp=(j - start_time) / duration,
                    pose=1,
                    hp_directions=1,
                    cxr=cxr,
                    cyr=cyr,
                )
            else:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    near=near,
                    far=far,
                    timestamp=(j - start_time) / duration,
                    pose=None,
                    hp_directions=None,
                    cxr=cxr,
                    cyr=cyr,
                )
            cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def read_colmap_cameras_immersive_v2(
    cam_extrinsics, cam_intrinsics, images_folder, near, far, start_time=0, duration=50
):
    cam_infos = []

    total_cam_name = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        total_cam_name.append(extr.name)

    sorted_total_cam_name_list = natsort.natsorted(total_cam_name)
    sorted_name_dict = {}
    for i in range(len(sorted_total_cam_name_list)):
        sorted_name_dict[sorted_total_cam_name_list[i]] = i  # map each cam with a number

    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec_2_rot_mat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        for j in range(start_time, start_time + int(duration)):
            # image_path = os.path.join(images_folder, os.path.basename(extr.name))
            # image_name = os.path.basename(image_path).split(".")[0]
            # image_path = image_path.replace("colmap_"+str(start_time), "colmap_{}".format(j), 1)

            parent_folder = os.path.dirname(images_folder)
            parent_folder = os.path.dirname(parent_folder)
            image_name = extr.name.split(".")[0]

            raw_video_folder = os.path.join(parent_folder, os.path.basename(image_name))

            image_path = os.path.join(raw_video_folder, str(j) + ".png")

            # image_path.replace("colmap_"+str(start_time), "colmap_{}".format(j), 1)

            # K = np.eye(3)
            # K[0, 0] = focal_length_x * 0.5
            # K[0, 2] = intr.params[2] * 0.5
            # K[1, 1] = focal_length_y * 0.5
            # K[1, 2] = intr.params[3] * 0.5

            cxr = (intr.params[2]) / width - 0.5
            cyr = (intr.params[3]) / height - 0.5

            hp_directions = 1.0

            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            image = Image.open(image_path)
            if j == start_time:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    near=near,
                    far=far,
                    timestamp=(j - start_time) / duration,
                    pose=1,
                    hp_directions=hp_directions,
                    cxr=cxr,
                    cyr=cyr,
                )
            else:
                cam_info = CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    near=near,
                    far=far,
                    timestamp=(j - start_time) / duration,
                    pose=None,
                    hp_directions=None,
                    cxr=cxr,
                    cyr=cyr,
                )
            cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


scene_load_type_callbacks = {
    "colmap": read_colmap_scene_info,
    "immersive": read_colmap_scene_info_immersive,
    "colmapmv": read_colmap_scene_info_mv,
    "blender": read_nerf_synthetic_info,
    "technicolor": read_colmap_scene_info_technicolor,
    "hyfluid": read_nerf_synthetic_info_hyfluid,
}
