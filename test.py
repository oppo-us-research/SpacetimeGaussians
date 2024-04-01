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
# ========================================================================================================
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the thirdparty/gaussian_splatting/LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import sys
import time
import warnings

from os import makedirs

import numpy as np
import scipy
import torch
import torchvision

from skimage.metrics import structural_similarity as sk_ssim
from torchvision.utils import save_image
from tqdm import tqdm

from helper_train import get_model, get_render_pipe, trb_function
from image_video_io import images_to_video
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams
from thirdparty.gaussian_splatting.helper3dg import get_test_parse
from thirdparty.gaussian_splatting.lpipsPyTorch import lpips
from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.loss_utils import ssim


warnings.filterwarnings("ignore")


# modified from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/render.py
# and https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py
def render_set(model_path, name, iteration, views, gaussians, pipe_args, background, rbf_base_function, rd_pipe):
    render, GRsetting, GRzer = get_render_pipe(rd_pipe)
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if gaussians.rgb_decoder is not None:
        gaussians.rgb_decoder.cuda()
        gaussians.rgb_decoder.eval()
    stats_dict = {}

    scales = gaussians.get_scaling

    scales_max = torch.amax(scales).item()
    scales_mean = torch.amin(scales).item()

    op = gaussians.get_opacity
    op_max = torch.amax(op).item()
    op_mean = torch.mean(op).item()

    stats_dict["scales_max"] = scales_max
    stats_dict["scales_mean"] = scales_mean

    stats_dict["op_max"] = op_max
    stats_dict["op_mean"] = op_mean

    stats_path = os.path.join(model_path, "stat_" + str(iteration) + ".json")
    with open(stats_path, "w") as fp:
        json.dump(stats_dict, fp, indent=True)

    psnrs = []
    lpipss = []
    lpipss_vggs = []

    full_dict = {}
    per_view_dict = {}
    ssims = []
    ssims_v2 = []
    scene_dir = model_path
    image_names = []
    times = []

    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}

    full_dict[scene_dir][iteration] = {}
    per_view_dict[scene_dir][iteration] = {}

    if rd_pipe == "train_ours_full":
        render, GRsetting, GRzer = get_render_pipe("test_ours_full")
    elif rd_pipe == "train_ours_lite":
        render, GRsetting, GRzer = get_render_pipe("test_ours_lite")
    else:
        render, GRsetting, GRzer = get_render_pipe(rd_pipe)

    for idx, view in enumerate(tqdm(views, desc="Rendering and metric progress")):
        rendering_pkg = render(
            view,
            gaussians,
            pipe_args,
            background,
            scaling_modifier=1.0,
            basic_function=rbf_base_function,
            GRsetting=GRsetting,
            GRzer=GRzer,
        )  # C x H x W
        rendering = rendering_pkg["render"]
        rendering = torch.clamp(rendering, 0, 1.0)
        gt = view.original_image[0:3, :, :].cuda().float()
        ssims.append(ssim(rendering.unsqueeze(0), gt.unsqueeze(0)))

        psnrs.append(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)))
        lpipss.append(lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type="alex"))
        lpipss_vggs.append(lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type="vgg"))

        render_numpy = rendering.permute(1, 2, 0).detach().cpu().numpy()
        gt_numpy = gt.permute(1, 2, 0).detach().cpu().numpy()
        ssim_v2 = sk_ssim(
            render_numpy, gt_numpy, channel_axis=-1, multichannel=True, data_range=gt_numpy.max() - gt_numpy.min()
        )
        ssims_v2.append(ssim_v2)

        save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))
        save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))
        image_names.append("{0:05d}".format(idx) + ".png")
    images_to_video(render_path, ".png", os.path.join(model_path, name + "_ours_{}.mp4".format(iteration)), fps=25)
    images_to_video(gts_path, ".png", os.path.join(model_path, name + "_gt_{}.mp4".format(iteration)), fps=25)

    for idx, view in enumerate(tqdm(views, desc="release gt images cuda memory for timing")):
        view.original_image = None  # .detach()
        torch.cuda.empty_cache()

    # start timing
    for _ in range(4):
        for idx, view in enumerate(tqdm(views, desc="timing ")):

            render_pack = render(
                view,
                gaussians,
                pipe_args,
                background,
                scaling_modifier=1.0,
                basic_function=rbf_base_function,
                GRsetting=GRsetting,
                GRzer=GRzer,
            )  # ["time"] # C x H x W
            duration = render_pack["duration"]
            if idx > 10:  # warm up
                times.append(duration)

    print(np.mean(np.array(times)))
    if len(views) > 0:
        full_dict[model_path][iteration].update(
            {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
                "SSIM_v2": torch.tensor(ssims_v2).mean().item(),
                "LPIPS_VGG": torch.tensor(lpipss_vggs).mean().item(),
                "times": torch.tensor(times).mean().item(),
            }
        )

        per_view_dict[model_path][iteration].update(
            {
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "LPIPS": {name: lpips for lpips, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                "SSIM_v2": {name: ssim2 for ssim2, name in zip(torch.tensor(ssims_v2).tolist(), image_names)},
                "LPIPS_VGG": {
                    name: lpips_vgg for lpips_vgg, name in zip(torch.tensor(lpipss_vggs).tolist(), image_names)
                },
            }
        )

        with open(model_path + "/" + str(iteration) + "_runtime_results.json", "w") as fp:
            json.dump(full_dict, fp, indent=True)

        with open(model_path + "/" + str(iteration) + "_runtime_per_view.json", "w") as fp:
            json.dump(per_view_dict, fp, indent=True)


# render free view
def render_set_no_gt(model_path, name, iteration, views, gaussians, pipe_args, background, rbf_base_function, rd_pipe):
    render, GRsetting, GRzer = get_render_pipe(rd_pipe)
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)
    if gaussians.rgb_decoder is not None:
        gaussians.rgb_decoder.cuda()
        gaussians.rgb_decoder.eval()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # C x H x W
        rendering = render(
            view,
            gaussians,
            pipe_args,
            background,
            scaling_modifier=1.0,
            basic_function=rbf_base_function,
            GRsetting=GRsetting,
            GRzer=GRzer,
        )
        rendered_image = rendering["render"]

        save_image(rendered_image, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))


@torch.no_grad()
def run_test(
    model_args: ModelParams,
    iteration: int,
    pipe_args: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    multi_view: bool,
    duration: int,
    rgb_function="rgbv1",
    rd_pipe="v2",
    loader="colmap",
):

    print("use model {}".format(model_args.model))
    GaussianModel = get_model(model_args.model)  # default, g_model, we are testing

    gaussians = GaussianModel(model_args.sh_degree, rgb_function)

    scene = Scene(
        model_args,
        gaussians,
        load_iteration=iteration,
        shuffle=False,
        multi_view=multi_view,
        duration=duration,
        loader=loader,
    )
    rbf_base_function = trb_function
    num_channels = 9
    bg_color = [0 for _ in range(num_channels)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if gaussians.ts is None:
        camera_list = scene.get_test_cameras()
        H, W = camera_list[0].image_height, camera_list[0].image_width
        gaussians.ts = torch.ones(1, 1, H, W).cuda()

    if not skip_test and not multi_view:
        print("rendering test set")
        render_set(
            model_args.model_path,
            "test",
            scene.loaded_iter,
            scene.get_test_cameras(),
            gaussians,
            pipe_args,
            background,
            rbf_base_function,
            rd_pipe,
        )
    if multi_view:
        print("rendering multi-view set no gt")
        render_set_no_gt(
            model_args.model_path,
            "mv",
            scene.loaded_iter,
            scene.get_test_cameras(),
            gaussians,
            pipe_args,
            background,
            rbf_base_function,
            rd_pipe,
        )


if __name__ == "__main__":

    args, model_args, pipe_args, multi_view = get_test_parse()
    run_test(
        model_args,
        args.test_iteration,
        pipe_args,
        args.skip_train,
        args.skip_test,
        multi_view,
        args.duration,
        rgb_function=args.rgb_function,
        rd_pipe=args.rd_pipe,
        loader=args.val_loader,
    )
