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
import sys

from argparse import ArgumentParser, Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.verify_llff = 0
        self.eval = False
        self.model = "gmodel"  #
        self.loader = "colmap"  #

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.feature_t_lr = 0.001
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005

        self.trbf_c_lr = 0.0001  #
        self.trbf_s_lr = 0.03
        self.trbf_scale_init = 0.0  #
        self.batch = 2
        self.move_lr = 3.5

        self.omega_lr = 0.0001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3_000
        self.opacity_reset_at = 10000
        self.densify_from_iter = 500
        self.densify_until_iter = 9000
        self.densify_grad_threshold = 0.0002
        self.rgb_lr = 0.0001
        self.densify_cnt = 6
        self.reg = 0
        self.lambda_reg = 0.0001
        self.shrinkscale = 2.0
        self.randomfeature = 0
        self.emstype = 0
        self.radials = 10.0
        self.new_ray_step = 2  #
        self.ems_start = 1600  # small for debug
        self.losstart = 200
        self.saveemppoints = 0  #
        self.prunebysize = 0
        self.ems_threshold = 0.6
        self.opacity_threshold = 0.005
        self.selectiveview = 0
        self.preprocess_points = 0
        self.fzrotit = 8001
        self.add_sph_points_scale = 0.8
        self.gnumlimit = 330000
        self.ray_end = 7.5
        self.ray_start = 0.7
        self.shuffle_ems = 1
        self.prev_path = "1"
        self.load_all = 0
        self.removescale = 5
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmd_line_string = sys.argv[1:]
    cfg_file_string = "Namespace()"
    args_cmdline = parser.parse_args(cmd_line_string)

    try:
        cfg_file_path = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfg_file_path)
        with open(cfg_file_path) as cfg_file:
            print("Config file found: {}".format(cfg_file_path))
            cfg_file_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfg_file = eval(cfg_file_string)

    merged_dict = vars(args_cfg_file).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
