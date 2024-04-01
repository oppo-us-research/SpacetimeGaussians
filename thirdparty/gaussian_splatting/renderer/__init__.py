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

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE ####################################
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################


import math
import time

import torch
import torch.nn.functional as F

from thirdparty.gaussian_splatting.scene.ours_full import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import (
    focal2fov,
    fov2focal,
    get_projection_matrix_cv,
)
from thirdparty.gaussian_splatting.utils.sh_utils import eval_sh


def train_ours_full(
    viewpoint_camera,
    gm: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    basic_function=None,
    GRsetting=None,
    GRzer=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screen_space_points = torch.zeros_like(gm.get_xyz, dtype=gm.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    point_times = torch.ones((gm.get_xyz.shape[0], 1), dtype=gm.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    try:
        screen_space_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tan_fov_x = math.tan(viewpoint_camera.FoVx * 0.5)
    tan_fov_y = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tan_fov_x=tan_fov_x,
        tan_fov_y=tan_fov_y,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        view_matrix=viewpoint_camera.world_view_transform,
        proj_matrix=viewpoint_camera.full_proj_transform,
        sh_degree=gm.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = gm.get_xyz
    means2D = screen_space_points
    point_opacity = gm.get_opacity

    trbf_center = gm.get_trbf_center
    trbf_scale = gm.get_trbf_scale

    trbf_distance_offset = viewpoint_camera.timestamp * point_times - trbf_center
    trbf_distance = trbf_distance_offset / torch.exp(trbf_scale)
    trbf_output = basic_function(trbf_distance)

    opacity = point_opacity * trbf_output  # - 0.5
    gm.trbf_output = trbf_output

    cov3D_precomp = None

    scales = gm.get_scaling
    shs = None
    tforpoly = trbf_distance_offset.detach()
    means3D = (
        means3D
        + gm._motion[:, 0:3] * tforpoly
        + gm._motion[:, 3:6] * tforpoly * tforpoly
        + gm._motion[:, 6:9] * tforpoly * tforpoly * tforpoly
    )
    rotations = gm.get_rotation(tforpoly)  # to try use
    colors_precomp = gm.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    rendered_image = gm.rgb_decoder(
        rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp
    )  # 1 , 3
    rendered_image = rendered_image.squeeze(0)
    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity": opacity,
        "depth": depth,
    }


def test_ours_full(
    viewpoint_camera,
    gm: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    basic_function=None,
    GRsetting=None,
    GRzer=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screen_space_points = torch.zeros_like(gm.get_xyz, dtype=gm.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    torch.cuda.synchronize()
    start_time = time.time()

    point_times = torch.ones((gm.get_xyz.shape[0], 1), dtype=gm.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    try:
        screen_space_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tan_fov_x = math.tan(viewpoint_camera.FoVx * 0.5)
    tan_fov_y = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tan_fov_x=tan_fov_x,
        tan_fov_y=tan_fov_y,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        view_matrix=viewpoint_camera.world_view_transform,
        proj_matrix=viewpoint_camera.full_proj_transform,
        sh_degree=gm.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = gm.get_xyz
    means2D = screen_space_points
    point_opacity = gm.get_opacity

    trbf_center = gm.get_trbf_center
    trbf_scale = gm.get_trbf_scale

    trbf_distance_offset = viewpoint_camera.timestamp * point_times - trbf_center
    trbf_distance = trbf_distance_offset / torch.exp(trbf_scale)
    trbf_output = basic_function(trbf_distance)

    opacity = point_opacity * trbf_output  # - 0.5
    gm.trbf_output = trbf_output

    cov3D_precomp = None

    scales = gm.get_scaling
    shs = None
    tforpoly = trbf_distance_offset.detach()
    means3D = (
        means3D
        + gm._motion[:, 0:3] * tforpoly
        + gm._motion[:, 3:6] * tforpoly * tforpoly
        + gm._motion[:, 6:9] * tforpoly * tforpoly * tforpoly
    )
    rotations = gm.get_rotation(tforpoly)  # to try use
    colors_precomp = gm.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    rendered_image = gm.rgb_decoder(
        rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp
    )  # 1 , 3
    rendered_image = rendered_image.squeeze(0)
    torch.cuda.synchronize()
    duration = time.time() - start_time

    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity": opacity,
        "depth": depth,
        "duration": duration,
    }


def test_ours_lite(
    viewpoint_camera,
    gm: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    basic_function=None,
    GRsetting=None,
    GRzer=None,
):

    screen_space_points = torch.zeros_like(gm.get_xyz, dtype=gm.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    torch.cuda.synchronize()
    start_time = time.time()

    tan_fov_x = math.tan(viewpoint_camera.FoVx * 0.5)
    tan_fov_y = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tan_fov_x=tan_fov_x,
        tan_fov_y=tan_fov_y,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        view_matrix=viewpoint_camera.world_view_transform,
        proj_matrix=viewpoint_camera.full_proj_transform,
        sh_degree=gm.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GRzer(raster_settings=raster_settings)

    tforpoly = viewpoint_camera.timestamp - gm.get_trbf_center
    rotations = gm.get_rotation(tforpoly)  # to try use
    colors_precomp = gm.get_features(tforpoly)

    means2D = screen_space_points

    cov3D_precomp = None

    shs = None

    rendered_image, radii = rasterizer(
        timestamp=viewpoint_camera.timestamp,
        trbf_center=gm.get_trbf_center,
        trbf_scale=gm.computed_trbf_scale,
        motion=gm._motion,
        means3D=gm.get_xyz,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=gm.computed_opacity,
        scales=gm.computed_scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    torch.cuda.synchronize()
    duration = time.time() - start_time
    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "duration": duration,
    }


def train_ours_lite(
    viewpoint_camera,
    gm: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    basic_function=None,
    GRsetting=None,
    GRzer=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screen_space_points = torch.zeros_like(gm.get_xyz, dtype=gm.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    point_times = torch.ones((gm.get_xyz.shape[0], 1), dtype=gm.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    try:
        screen_space_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tan_fov_x = math.tan(viewpoint_camera.FoVx * 0.5)
    tan_fov_y = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tan_fov_x=tan_fov_x,
        tan_fov_y=tan_fov_y,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        view_matrix=viewpoint_camera.world_view_transform,
        proj_matrix=viewpoint_camera.full_proj_transform,
        sh_degree=gm.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = gm.get_xyz
    means2D = screen_space_points
    point_opacity = gm.get_opacity

    trbf_center = gm.get_trbf_center
    trbf_scale = gm.get_trbf_scale

    trbf_distance_offset = viewpoint_camera.timestamp * point_times - trbf_center
    trbf_distance = trbf_distance_offset / torch.exp(trbf_scale)
    trbf_output = basic_function(trbf_distance)

    opacity = point_opacity * trbf_output  # - 0.5
    gm.trbf_output = trbf_output

    cov3D_precomp = None

    scales = gm.get_scaling

    shs = None
    tforpoly = trbf_distance_offset.detach()
    means3D = (
        means3D
        + gm._motion[:, 0:3] * tforpoly
        + gm._motion[:, 3:6] * tforpoly * tforpoly
        + gm._motion[:, 6:9] * tforpoly * tforpoly * tforpoly
    )

    rotations = gm.get_rotation(tforpoly)  # to try use
    colors_precomp = gm.get_features(tforpoly)

    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity": opacity,
        "depth": depth,
    }
