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
import shutil
import sys
import time
import uuid

from argparse import ArgumentParser, Namespace
from random import randint

import cv2
import numpy as np
import torch

from tqdm import tqdm

from thirdparty.gaussian_splatting.arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from thirdparty.gaussian_splatting.utils.general_utils import safe_state


def get_parser():
    parser = ArgumentParser(description="Training script parameters")
    mp = ModelParams(parser)
    op = OptimizationParams(parser)  # we put more parameters in optimization params, just for convenience.
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6029)
    parser.add_argument("--debug_from", type=int, default=-2)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)

    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 12_000, 25_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 12_000, 30_000])

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--densify", type=int, default=1, help="densify =1, we control points on N3d dataset")
    parser.add_argument("--duration", type=int, default=5, help="5 debug , 50 used")
    parser.add_argument("--basic_function", type=str, default="gaussian")
    parser.add_argument("--rgb_function", type=str, default="rgbv1")
    parser.add_argument("--rd_pipe", type=str, default="v2", help="render pipeline")
    parser.add_argument("--config_path", type=str, default="None")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # incase we provide config file not directly pass to the file
    if os.path.exists(args.config_path) and args.config_path != "None":
        print("overload config from " + args.config_path)
        config = json.load(open(args.config_path))
        for k in config.keys():
            try:
                value = getattr(args, k)
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.config_path)
    else:
        raise ValueError("config file not exist or not provided")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    return args, mp.extract(args), op.extract(args), pp.extract(args)


def get_render_parts(render_pkg):
    return render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


def get_test_parse():
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--test_iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--multi_view", action="store_true")
    parser.add_argument("--duration", default=50, type=int)
    parser.add_argument("--rgb_function", type=str, default="rgbv1")
    parser.add_argument("--rd_pipe", type=str, default="v3", help="render pipeline")
    parser.add_argument("--val_loader", type=str, default="colmap")
    parser.add_argument("--config_path", type=str, default="1")

    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # config_path
    safe_state(args.quiet)

    multi_view = True if args.val_loader.endswith("mv") else False

    if os.path.exists(args.config_path) and args.config_path != "None":
        print("overload config from " + args.config_path)
        config = json.load(open(args.config_path))
        for k in config.keys():
            try:
                value = getattr(args, k)
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.config_path)
        print("args: " + str(args))

        return args, model.extract(args), pipeline.extract(args), multi_view


def get_colmap_single_n3d(folder, offset):

    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    db_file = os.path.join(folder, "input.db")
    input_image_folder = os.path.join(folder, "input")
    distorted_model = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manual_input_folder = os.path.join(folder, "manual")
    if not os.path.exists(distorted_model):
        os.makedirs(distorted_model)

    feature_extract = f"colmap feature_extractor --database_path {db_file} --image_path {input_image_folder}"

    exit_code = os.system(feature_extract)
    if exit_code != 0:
        exit(exit_code)

    feature_matcher = f"colmap exhaustive_matcher --database_path {db_file}"
    exit_code = os.system(feature_matcher)
    if exit_code != 0:
        exit(exit_code)

    # threshold is from   https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/scripts/local_colmap_and_resize.sh#L62
    tri_and_map = (
        "colmap point_triangulator --database_path "
        + db_file
        + " --image_path "
        + input_image_folder
        + " --output_path "
        + distorted_model
        + " --input_path "
        + manual_input_folder
        + " --Mapper.ba_global_function_tolerance=0.000001"
    )

    exit_code = os.system(tri_and_map)
    if exit_code != 0:
        exit(exit_code)
    print(tri_and_map)

    img_undist_cmd = (
        "colmap"
        + " image_undistorter --image_path "
        + input_image_folder
        + " --input_path "
        + distorted_model
        + " --output_path "
        + folder
        + " --output_type COLMAP"
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    remove_input = "rm -r " + input_image_folder
    exit_code = os.system(remove_input)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == "0":
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def get_colmap_single_im_undistort(folder, offset):

    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    db_file = os.path.join(folder, "input.db")
    input_image_folder = os.path.join(folder, "input")
    distorted_model = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manual_input_folder = os.path.join(folder, "manual")
    if not os.path.exists(distorted_model):
        os.makedirs(distorted_model)

    feature_extract = (
        "colmap feature_extractor SiftExtraction.max_image_size 6000 --database_path "
        + db_file
        + " --image_path "
        + input_image_folder
    )

    exit_code = os.system(feature_extract)
    if exit_code != 0:
        exit(exit_code)

    feature_matcher = "colmap exhaustive_matcher --database_path " + db_file
    exit_code = os.system(feature_matcher)
    if exit_code != 0:
        exit(exit_code)

    tri_and_map = (
        "colmap point_triangulator --database_path "
        + db_file
        + " --image_path "
        + input_image_folder
        + " --output_path "
        + distorted_model
        + " --input_path "
        + manual_input_folder
        + " --Mapper.ba_global_function_tolerance=0.000001"
    )

    exit_code = os.system(tri_and_map)
    if exit_code != 0:
        exit(exit_code)
    print(tri_and_map)

    img_undist_cmd = (
        "colmap"
        + " image_undistorter --image_path "
        + input_image_folder
        + " --input_path "
        + distorted_model
        + " --output_path "
        + folder
        + " --output_type COLMAP "
    )  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    remove_input = "rm -r " + input_image_folder
    exit_code = os.system(remove_input)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == "0":
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def get_colmap_single_im_distort(folder, offset):

    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    db_file = os.path.join(folder, "input.db")
    input_image_folder = os.path.join(folder, "input")
    distorted_model = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manual_input_folder = os.path.join(folder, "manual")
    if not os.path.exists(distorted_model):
        os.makedirs(distorted_model)

    feature_extract = (
        "colmap feature_extractor SiftExtraction.max_image_size 6000 --database_path "
        + db_file
        + " --image_path "
        + input_image_folder
    )

    exit_code = os.system(feature_extract)
    if exit_code != 0:
        exit(exit_code)

    feature_matcher = "colmap exhaustive_matcher --database_path " + db_file
    exit_code = os.system(feature_matcher)
    if exit_code != 0:
        exit(exit_code)

    tri_and_map = (
        "colmap point_triangulator --database_path "
        + db_file
        + " --image_path "
        + input_image_folder
        + " --output_path "
        + distorted_model
        + " --input_path "
        + manual_input_folder
        + " --Mapper.ba_global_function_tolerance=0.000001"
    )

    exit_code = os.system(tri_and_map)
    if exit_code != 0:
        exit(exit_code)
    print(tri_and_map)

    img_undist_cmd = (
        "colmap"
        + " image_undistorter --image_path "
        + input_image_folder
        + " --input_path "
        + distorted_model
        + " --output_path "
        + folder
        + " --output_type COLMAP "
    )  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    remove_input = "rm -r " + input_image_folder
    exit_code = os.system(remove_input)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == "0":
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def get_colmap_single_techni(folder, offset):

    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    db_file = os.path.join(folder, "input.db")
    input_image_folder = os.path.join(folder, "input")
    distorted_model = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manual_input_folder = os.path.join(folder, "manual")
    if not os.path.exists(distorted_model):
        os.makedirs(distorted_model)

    feature_extract = "colmap feature_extractor --database_path " + db_file + " --image_path " + input_image_folder

    exit_code = os.system(feature_extract)
    if exit_code != 0:
        exit(exit_code)

    feature_matcher = "colmap exhaustive_matcher --database_path " + db_file
    exit_code = os.system(feature_matcher)
    if exit_code != 0:
        exit(exit_code)

    tri_and_map = (
        "colmap point_triangulator --database_path "
        + db_file
        + " --image_path "
        + input_image_folder
        + " --output_path "
        + distorted_model
        + " --input_path "
        + manual_input_folder
        + " --Mapper.ba_global_function_tolerance=0.000001"
    )

    exit_code = os.system(tri_and_map)
    if exit_code != 0:
        exit(exit_code)
    print(tri_and_map)

    img_undist_cmd = (
        "colmap"
        + " image_undistorter --image_path "
        + input_image_folder
        + " --input_path "
        + distorted_model
        + " --output_path "
        + folder
        + " --output_type COLMAP "
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == "0":
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    return
