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

import json
import os
import random
import sys
import time
import uuid

from argparse import Namespace
from random import randint

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from helper_train import (
    control_gaussians,
    get_loss,
    get_model,
    get_render_pipe,
    reload_helper,
    trb_function,
)
from thirdparty.gaussian_splatting.helper3dg import get_parser, get_render_parts
from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.loss_utils import (
    l1_loss,
    l2_loss,
    relative_loss,
    ssim,
)


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def train(
    model_args,
    optim_args,
    pipe_args,
    testing_iterations,
    saving_iterations,
    debug_from,
    densify=0,
    duration=50,
    rgb_function="rgbv1",
    rd_pipe="v2",
):
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = prepare_output_and_logger(model_args)
    first_iter = 0
    render, GRsetting, GRzer = get_render_pipe(rd_pipe)

    print("use model {}".format(model_args.model))
    GaussianModel = get_model(model_args.model)  # g_model, g_model_rgb_only

    # trbf means Temporal Radial Basis Function in the paper
    # the trbf_center µ^τ_i is the temporal center, trbf_scale s^τ_i is temporal scaling factor
    gaussians = GaussianModel(model_args.sh_degree, rgb_function)
    gaussians.trbf_scale_init = -1 * optim_args.trbf_scale_init
    gaussians.preprocess_points = optim_args.preprocess_points
    gaussians.add_sph_points_scale = optim_args.add_sph_points_scale
    gaussians.ray_start = optim_args.ray_start

    trbf_base_function = trb_function
    scene = Scene(model_args, gaussians, duration=duration, loader=model_args.loader)

    current_xyz = gaussians._xyz
    # z wrong... # ???
    max_x, max_y, max_z = torch.amax(current_xyz[:, 0]), torch.amax(current_xyz[:, 1]), torch.amax(current_xyz[:, 2])
    min_x, min_y, min_z = torch.amin(current_xyz[:, 0]), torch.amin(current_xyz[:, 1]), torch.amin(current_xyz[:, 2])

    if os.path.exists(optim_args.prev_path):
        print("load from " + optim_args.prev_path)
        reload_helper(gaussians, optim_args, max_x, max_y, max_z, min_x, min_y, min_z)

    max_bounds = [max_x, max_y, max_z]
    min_bounds = [min_x, min_y, min_z]

    gaussians.training_setup(optim_args)

    num_channel = 9

    bg_color = [1, 1, 1] if model_args.white_background else [0 for i in range(num_channel)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    # if freeze != 1:
    first_iter = 0
    progress_bar = tqdm(range(first_iter, optim_args.iterations), desc="Training progress")
    first_iter += 1

    flag = 0

    depth_dict = {}

    if optim_args.batch > 1:
        train_camera_list = scene.get_train_cameras().copy()
        train_cam_dict = {}
        for i in range(duration):  # 0 to 4, -> (0.0, to 0.8)
            train_cam_dict[i] = [cam for cam in train_camera_list if cam.timestamp == i / duration]

    if gaussians.ts is None:
        H, W = train_camera_list[0].image_height, train_camera_list[0].image_width
        gaussians.ts = torch.ones(1, 1, H, W).cuda()

    scene.record_points(0, "start training")

    print("after scene.record_points : ", gaussians._xyz.shape[0])

    flag_ems = 0
    ems_cnt = 0
    loss_dict = {}
    ssim_dict = {}
    depth_dict = {}
    valid_depth_dict = {}
    ems_start_from_iterations = optim_args.ems_start

    with torch.no_grad():
        time_index = 0  # 0 to 49
        viewpoint_set = train_cam_dict[time_index]
        for viewpoint_cam in viewpoint_set:
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe_args,
                background,
                override_color=None,
                basic_function=trbf_base_function,
                GRsetting=GRsetting,
                GRzer=GRzer,
            )

            _, depthH, depthW = render_pkg["depth"].shape
            border_H = int(depthH / 2)
            border_W = int(depthW / 2)

            mid_h = int(viewpoint_cam.image_height / 2)
            mid_w = int(viewpoint_cam.image_width / 2)

            depth = render_pkg["depth"]
            select_mask = depth != 15.0
            initial_image = render_pkg["render"]

            valid_depth_dict[viewpoint_cam.image_name] = torch.median(depth[select_mask]).item()
            depth_dict[viewpoint_cam.image_name] = torch.amax(depth[select_mask]).item()
            save_image(initial_image, os.path.join(scene.model_path, f"initial_render_{viewpoint_cam.image_name}.png"))

    print("after depth_dict[viewpoint_cam.image_name] : ", gaussians._xyz.shape[0])

    if args.loader != "hyfluid" and (densify == 1 or densify == 2):
        z_mask = gaussians._xyz[:, 2] < 4.5
        gaussians.prune_points(z_mask)
        torch.cuda.empty_cache()

    selected_length = 2
    laster_ems = 0

    print("before training iteration: ", gaussians._xyz.shape[0])

    for iteration in range(first_iter, optim_args.iterations + 1):
        if iteration == optim_args.ems_start:
            flag_ems = 1  # start ems

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if (iteration - 1) == debug_from:
            pipe_args.debug = True
        if gaussians.rgb_decoder is not None:
            gaussians.rgb_decoder.train()

        if optim_args.batch > 1:
            gaussians.zero_gradient_cache()
            time_index = randint(0, duration - 1)  # 0 to 49
            viewpoint_set = train_cam_dict[time_index]
            cam_index = random.sample(viewpoint_set, optim_args.batch)

            for i in range(optim_args.batch):
                viewpoint_cam = cam_index[i]
                render_pkg = render(
                    viewpoint_cam,
                    gaussians,
                    pipe_args,
                    background,
                    override_color=None,
                    basic_function=trbf_base_function,
                    GRsetting=GRsetting,
                    GRzer=GRzer,
                )
                image, viewspace_point_tensor, visibility_filter, radii = get_render_parts(render_pkg)
                gt_image = viewpoint_cam.original_image.float().cuda()

                if optim_args.reg == 2:
                    Ll1 = l2_loss(image, gt_image)
                    loss = Ll1
                elif optim_args.reg == 3:
                    Ll1 = relative_loss(image, gt_image)
                    loss = Ll1
                else:
                    Ll1 = l1_loss(image, gt_image)
                    loss = get_loss(optim_args, Ll1, ssim, image, gt_image, gaussians, radii)

                if flag_ems == 1:
                    if viewpoint_cam.image_name not in loss_dict:
                        loss_dict[viewpoint_cam.image_name] = loss.item()
                        ssim_dict[viewpoint_cam.image_name] = ssim(
                            image.clone().detach(), gt_image.clone().detach()
                        ).item()

                loss.backward()
                gaussians.cache_gradient()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if flag_ems == 1 and len(loss_dict.keys()) == len(viewpoint_set):
                # sort dict by value
                # ssim_dict loss_dict
                ordered_loss_dict = sorted(ssim_dict.items(), key=lambda item: item[1], reverse=False)
                flag_ems = 2
                select_views_list = []
                select_views = {}
                for idx, pair in enumerate(ordered_loss_dict):
                    viewname, loss_score = pair
                    ssim_score = ssim_dict[viewname]
                    if ssim_score < 0.91:  # avoid large ssim
                        select_views_list.append((viewname, "rk" + str(idx) + "_ssim" + str(ssim_score)[0:4]))
                if len(select_views_list) < 2:
                    select_views = []
                else:
                    select_views_list = select_views_list[:2]
                    for v in select_views_list:
                        select_views[v[0]] = v[1]

                selected_length = len(select_views)

            iter_end.record()
            gaussians.set_batch_gradient(optim_args.batch)
            # note we retrieve the correct gradient except the mask
        else:
            raise NotImplementedError("Batch size 1 is not supported")

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == optim_args.iterations:
                progress_bar.close()

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe_args, background),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification and pruning here

            if iteration < optim_args.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            flag = control_gaussians(
                optim_args,
                gaussians,
                densify,
                iteration,
                scene,
                visibility_filter,
                radii,
                viewspace_point_tensor,
                flag,
                train_camera_with_distance=None,
                max_bounds=max_bounds,
                min_bounds=min_bounds,
            )

            # guided sampling step
            if (
                iteration > ems_start_from_iterations
                and flag_ems == 2
                and ems_cnt < selected_length
                and viewpoint_cam.image_name in select_views
                and (iteration - laster_ems > 100)
            ):
                # ["camera_0002"] :#select_views :  #["camera_0002"]:
                select_views.pop(viewpoint_cam.image_name)  # remove sampled cameras
                ems_cnt += 1
                laster_ems = iteration
                ssim_current = ssim(image.detach(), gt_image.detach()).item()
                scene.record_points(iteration, "ssim_" + str(ssim_current))
                # some scenes' structure is already good, no need to add more points
                if ssim_current < 0.88:
                    image_adjust = image / (torch.mean(image) + 0.01)  #
                    gt_adjust = gt_image / (torch.mean(gt_image) + 0.01)
                    diff = torch.abs(image_adjust - gt_adjust)
                    diff = torch.sum(diff, dim=0)  # h, w
                    diff_sorted, _ = torch.sort(diff.reshape(-1))
                    num_pixels = diff.shape[0] * diff.shape[1]
                    threshold = diff_sorted[int(num_pixels * optim_args.ems_threshold)].item()
                    out_mask = diff > threshold  #
                    kh, kw = 16, 16  # kernel size
                    dh, dw = 16, 16  # stride
                    # compute padding
                    ideal_h, ideal_w = (
                        int(image.shape[1] / dh + 1) * kw,
                        int(image.shape[2] / dw + 1) * kw,
                    )

                    out_mask = torch.nn.functional.pad(
                        out_mask,
                        (0, ideal_w - out_mask.shape[1], 0, ideal_h - out_mask.shape[0]),
                        mode="constant",
                        value=0,
                    )
                    patches = out_mask.unfold(0, kh, dh).unfold(1, kw, dw)
                    dummy_patch = torch.ones_like(patches)
                    patches_sum = patches.sum(dim=(2, 3))
                    patches_mask = patches_sum > kh * kh * 0.85
                    patches_mask = patches_mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, kh, kh).float()
                    patches = dummy_patch * patches_mask

                    depth = render_pkg["depth"]
                    depth = depth.squeeze(0)
                    # compute padding for depth
                    ideal_depth_h, ideal_depth_w = (
                        int(depth.shape[0] / dh + 1) * kw,
                        int(depth.shape[1] / dw + 1) * kw,
                    )

                    depth = torch.nn.functional.pad(
                        depth,
                        (0, ideal_depth_w - depth.shape[1], 0, ideal_depth_h - depth.shape[0]),
                        mode="constant",
                        value=0,
                    )

                    depth_patches = depth.unfold(0, kh, dh).unfold(1, kw, dw)
                    dummy_depth_patches = torch.ones_like(depth_patches)
                    a, b, c, d = depth_patches.shape
                    depth_patches = depth_patches.reshape(a, b, c * d)
                    median_depth_patch = torch.median(depth_patches, dim=(2))[0]
                    depth_patches = dummy_depth_patches * (median_depth_patch.unsqueeze(2).unsqueeze(3))
                    unfold_depth_shape = dummy_depth_patches.size()
                    output_depth_h = unfold_depth_shape[0] * unfold_depth_shape[2]
                    output_depth_w = unfold_depth_shape[1] * unfold_depth_shape[3]

                    patches_depth_orig = depth_patches.view(unfold_depth_shape)
                    patches_depth_orig = patches_depth_orig.permute(0, 2, 1, 3).contiguous()
                    # 1 for error, 0 for no error
                    patches_depth = patches_depth_orig.view(output_depth_h, output_depth_w).float()

                    depth = patches_depth[: render_pkg["depth"].shape[1], : render_pkg["depth"].shape[2]]
                    depth = depth.unsqueeze(0)

                    mid_patch = torch.ones_like(patches)

                    for i in range(0, kh, 2):
                        for j in range(0, kw, 2):
                            mid_patch[:, :, i, j] = 0.0

                    center_patches = patches * mid_patch

                    unfold_shape = patches.size()
                    patches_orig = patches.view(unfold_shape)
                    center_patches_orig = center_patches.view(unfold_shape)

                    output_h = unfold_shape[0] * unfold_shape[2]
                    output_w = unfold_shape[1] * unfold_shape[3]
                    patches_orig = patches_orig.permute(0, 2, 1, 3).contiguous()
                    center_patches_orig = center_patches_orig.permute(0, 2, 1, 3).contiguous()
                    # H * W  mask, # 1 for error, 0 for no error
                    center_mask = center_patches_orig.view(output_h, output_w).float()
                    center_mask = center_mask[: image.shape[1], : image.shape[2]]  # reverse back

                    # H * W  mask, # 1 for error, 0 for no error
                    error_mask = patches_orig.view(output_h, output_w).float()
                    error_mask = error_mask[: image.shape[1], : image.shape[2]]  # reverse back

                    H, W = center_mask.shape

                    offset_H = int(H / 10)
                    offset_W = int(W / 10)

                    center_mask[0:offset_H, :] = 0.0
                    center_mask[:, 0:offset_W] = 0.0

                    center_mask[-offset_H:, :] = 0.0
                    center_mask[:, -offset_W:] = 0.0

                    depth = render_pkg["depth"]
                    depth_map = torch.cat((depth, depth, depth), dim=0)
                    invalid_depth_mask = depth == 15.0

                    path_dir = scene.model_path + "/ems_" + str(ems_cnt - 1)
                    if not os.path.exists(path_dir):
                        os.makedirs(path_dir)

                    depth_map = depth_map / torch.amax(depth_map)
                    invalid_depth_map = torch.cat(
                        (invalid_depth_mask, invalid_depth_mask, invalid_depth_mask), dim=0
                    ).float()

                    save_image(gt_image, os.path.join(path_dir, "gt" + str(iteration) + ".png"))
                    save_image(image, os.path.join(path_dir, "render" + str(iteration) + ".png"))
                    save_image(depth_map, os.path.join(path_dir, "depth" + str(iteration) + ".png"))
                    save_image(invalid_depth_map, os.path.join(path_dir, "invalid_depth" + str(iteration) + ".png"))

                    bad_indices = center_mask.nonzero()
                    diff_sorted, _ = torch.sort(depth.reshape(-1))
                    N = diff_sorted.shape[0]
                    median_depth = int(0.7 * N)
                    median_depth = diff_sorted[median_depth]

                    depth = torch.where(depth > median_depth, depth, median_depth)

                    total_N_new_points = gaussians.add_gaussians(
                        bad_indices,
                        viewpoint_cam,
                        depth,
                        gt_image,
                        new_ray_step=optim_args.new_ray_step,
                        ray_end=optim_args.ray_end,
                        depth_max=depth_dict[viewpoint_cam.image_name],
                        shuffle=(optim_args.shuffle_ems != 0),
                    )

                    gt_image = gt_image * error_mask
                    image = render_pkg["render"] * error_mask

                    scene.record_points(iteration, "after add points by uv")

                    save_image(gt_image, os.path.join(path_dir, "masked_gt" + str(iteration) + ".png"))
                    save_image(image, os.path.join(path_dir, "masked_render" + str(iteration) + ".png"))
                    visibility_filter = torch.cat((visibility_filter, torch.zeros(total_N_new_points).cuda(0)), dim=0)
                    visibility_filter = visibility_filter.bool()
                    radii = torch.cat((radii, torch.zeros(total_N_new_points).cuda(0)), dim=0)
                    viewspace_point_tensor = torch.cat(
                        (viewspace_point_tensor, torch.zeros(total_N_new_points, 3).cuda(0)), dim=0
                    )

            # Optimizer step
            if iteration < optim_args.iterations:
                gaussians.optimizer.step()

                gaussians.optimizer.zero_grad(set_to_none=True)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    render_func,
    render_args,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.get_test_cameras()},
            {
                "name": "train",
                "cameras": [
                    scene.get_train_cameras()[idx % len(scene.get_train_cameras())] for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(render_func(viewpoint, scene.gaussians, *render_args)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":

    args, lp_extract, op_extract, pp_extract = get_parser()
    train(
        lp_extract,
        op_extract,
        pp_extract,
        args.test_iterations,
        args.save_iterations,
        args.debug_from,
        densify=args.densify,
        duration=args.duration,
        rgb_function=args.rgb_function,
        rd_pipe=args.rd_pipe,
    )

    # All done
    print("\nTraining complete.")
