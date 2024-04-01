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

import argparse
import glob
import os
import pickle
import shutil
import struct

import cv2
import natsort
import numpy as np
import tqdm

from scipy.spatial.transform import Rotation

from script.pre_n3d import extract_frames
from thirdparty.colmap.pre_colmap import COLMAPDatabase
from thirdparty.gaussian_splatting.helper3dg import get_colmap_single_im_distort
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov, fov2focal
from thirdparty.gaussian_splatting.utils.my_utils import (
    pose_to_w2c_matrixes,
    qvec_2_rot_mat,
    rot_mat_2_qvec,
)


SCALE_DICT = {}

# SCALE_DICT["01_Welder_S11"] = 0.35
# ["04_Truck", "09_Alexa", "10_Alexa", "11_Alexa", "12_Cave"]
immersive_seven = ["01_Welder", "02_Flames", "04_Truck", "09_Alexa", "10_Alexa", "11_Alexa", "12_Cave"]
immersive_scale_dict = {}
immersive_scale_dict["01_Welder"] = 0.36
immersive_scale_dict["02_Flames"] = 0.35
immersive_scale_dict["04_Truck"] = 0.36
immersive_scale_dict["09_Alexa"] = 0.36
immersive_scale_dict["10_Alexa"] = 0.36
immersive_scale_dict["11_Alexa"] = 0.36
immersive_scale_dict["12_Cave"] = 0.36

for scene in immersive_seven:
    immersive_scale_dict[scene + "_dist"] = immersive_scale_dict[scene]
    SCALE_DICT[scene + "_dist"] = immersive_scale_dict[scene]
    # immersive_scale_dict[scene]  # to be checked with large scale


def convert_model_to_db_files(path, offset=0, scale=1.0, remove_everything_except_input=False):

    project_folder = os.path.join(path, "colmap_" + str(offset))
    manual_folder = os.path.join(project_folder, "manual")

    if os.path.exists(project_folder) and remove_everything_except_input:
        print("already exists colmap folder, better remove it and create a new one")
        input_folder = os.path.join(project_folder, "input")
        # remove everything except input folder
        for file in os.listdir(project_folder):
            if file == "input":
                continue
            file_path = os.path.join(project_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    if not os.path.exists(manual_folder):
        os.makedirs(manual_folder)

    save_txt = os.path.join(manual_folder, "images.txt")
    save_camera = os.path.join(manual_folder, "cameras.txt")
    save_points = os.path.join(manual_folder, "points3D.txt")
    image_txt_list = []
    camera_txt_list = []
    if os.path.exists(os.path.join(project_folder, "input.db")):
        os.remove(os.path.join(project_folder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(project_folder, "input.db"))

    db.create_tables()

    import json

    with open(os.path.join(video + "models.json"), "r") as f:
        meta = json.load(f)

    for idx, camera in enumerate(meta):
        camera_name = camera["name"]  # camera_0001
        view = camera

        focal_length = camera["focal_length"]
        width, height = camera["width"], camera["height"]
        principle_point = [0, 0]
        principle_point[0] = view["principal_point"][0]
        principle_point[1] = view["principal_point"][1]

        distort1 = view["radial_distortion"][0]
        distort2 = view["radial_distortion"][1]
        distort3 = 0
        distort4 = 0  # view['radial_distortion'][3]

        R = Rotation.from_rotvec(view["orientation"]).as_matrix()
        t = np.array(view["position"])[:, np.newaxis]
        w2c = np.concatenate((R, -np.dot(R, t)), axis=1)

        colmapR = w2c[:3, :3]
        T = w2c[:3, 3]

        K = np.array([[focal_length, 0, principle_point[0]], [0, focal_length, principle_point[1]], [0, 0, 1]])
        Knew = K.copy()

        Knew[0, 0] = K[0, 0] * float(scale)
        Knew[1, 1] = K[1, 1] * float(scale)
        Knew[0, 2] = view["principal_point"][0]  # width * 0.5 #/ 2
        Knew[1, 2] = view["principal_point"][1]  # height * 0.5 #/ 2

        # transformation = np.array([[2,   0.0, 0.5],
        #                            [0.0, 2,   0.5],
        #                            [0.0, 0.0, 1.0]])
        # Knew = np.dot(transformation, Knew)

        new_focal_x = Knew[0, 0]
        new_focal_y = Knew[1, 1]
        new_cx = Knew[0, 2]
        new_cy = Knew[1, 2]

        colmapQ = rot_mat_2_qvec(colmapR)

        image_id = str(idx + 1)
        camera_id = image_id
        png_name = camera_name + ".png"

        line = image_id + " "

        for j in range(4):
            line += str(colmapQ[j]) + " "
        for j in range(3):
            line += str(T[j]) + " "
        line = line + camera_id + " " + png_name + "\n"
        empty_line = "\n"
        image_txt_list.append(line)
        image_txt_list.append(empty_line)

        new_width = width
        new_height = height
        params = np.array(
            (
                new_focal_x,
                new_focal_y,
                new_cx,
                new_cy,
            )
        )

        camera_id = db.add_camera(1, new_width, new_height, params)
        # RADIAL_FISHEYE                                                                                 # width and height
        #
        # camera_line = str(i+1) + " " + "PINHOLE " + str(width) +  " " + str(height) + " " + str(focal_length) + " " + str(focal_length) + " " + str(W//2) + " " + str(H//2) + "\n"

        camera_line = (
            str(idx + 1)
            + " "
            + "PINHOLE "
            + str(new_width)
            + " "
            + str(new_height)
            + " "
            + str(new_focal_x)
            + " "
            + str(new_focal_y)
            + " "
            + str(new_cx)
            + " "
            + str(new_cy)
            + "\n"
        )
        camera_txt_list.append(camera_line)
        image_id = db.add_image(
            png_name,
            camera_id,
            prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])),
            prior_t=np.array((T[0], T[1], T[2])),
            image_id=idx + 1,
        )
        db.commit()
        print("committed one")
    db.close()

    with open(save_txt, "w") as f:
        for line in image_txt_list:
            f.write(line)
    with open(save_camera, "w") as f:
        for line in camera_txt_list:
            f.write(line)
    with open(save_points, "w") as f:
        pass


# https://github.com/Synthesis-AI-Dev/fisheye-distortion
def get_distorted_flow(
    img: np.ndarray,
    cam_intr: np.ndarray,
    dist_coeff: np.ndarray,
    mode: str,
    crop_output: bool = True,
    crop_type: str = "corner",
    scale: float = 2,
    cx_offset=None,
    cy_offset=None,
    knew=None,
):

    assert cam_intr.shape == (3, 3)
    assert dist_coeff.shape == (4,)

    im_shape = img.shape
    if len(im_shape) == 3:
        h, w, chan = im_shape
    elif len(im_shape) == 2:
        h, w = im_shape
        chan = 1
    else:
        raise RuntimeError(f"Image has unsupported shape: {im_shape}. Valid shapes: (H, W), (H, W, N)")

    im_dtype = img.dtype
    dstW = int(w)
    dstH = int(h)

    # Get array of pixel coords
    xs = np.arange(dstW)
    ys = np.arange(dstH)

    xs = xs  # - 0.5 # + cx_offset / 2
    ys = ys  # - 0.5 # + cy_offset / 2

    xv, yv = np.meshgrid(xs, ys)
    img_pts = np.stack((xv, yv), axis=2)  # shape (H, W, 2)
    img_pts = img_pts.reshape((-1, 1, 2)).astype(np.float32)  # shape: (N, 1, 2), in undistorted image coordinate

    undistorted_px = cv2.fisheye.undistortPoints(img_pts, cam_intr, dist_coeff, None, knew)  # shape: (N, 1, 2)

    undistorted_px = undistorted_px.reshape((dstH, dstW, 2))  # Shape: (H, W, 2)
    undistorted_px = np.flip(undistorted_px, axis=2)  # flip x, y coordinates of the points as cv2 is height first

    undistorted_px[:, :, 0] = undistorted_px[:, :, 0]  # +  0.5*cy_offset #- 0.25*cy_offset #original_x (0, 1)
    undistorted_px[:, :, 1] = undistorted_px[:, :, 1]  # +  0.5*cy_offset #- 0.25*cx_offset #original_y (0, 1)

    undistorted_px[:, :, 0] = undistorted_px[:, :, 0] / (h - 1)  # (h-1) #original_x (0, 1)
    undistorted_px[:, :, 1] = undistorted_px[:, :, 1] / (w - 1)  # (w-1) #original_y (0, 1)

    undistorted_px = 2 * (undistorted_px - 0.5)  # to -1 to 1 for grid_sample

    undistorted_px[:, :, 0] = undistorted_px[:, :, 0]  # original_x (0, 1)
    undistorted_px[:, :, 1] = undistorted_px[:, :, 1]  # original_y (0, 1)

    undistorted_px = undistorted_px[:, :, ::-1]  # yx to xy for grid sample
    return undistorted_px


def image_undistort(video, offset_list=[0], focal_scale=1.0, fix_focal=None):
    import json
    import os

    import cv2
    import numpy as np

    with open(os.path.join(video + "models.json"), "r") as f:
        meta = json.load(f)

    for idx, camera in enumerate(meta):
        folder = camera["name"]  # camera_0001
        view = camera
        intrinsics = np.array(
            [
                [view["focal_length"], 0.0, view["principal_point"][0]],
                [0.0, view["focal_length"], view["principal_point"][1]],
                [0.0, 0.0, 1.0],
            ]
        )
        dis_cef = np.zeros((4))

        dis_cef[:2] = np.array(view["radial_distortion"])[:2]
        print("done one camera")
        map1, map2 = None, None
        for offset in offset_list:
            video_folder = os.path.join(video, folder)
            image_path = os.path.join(video_folder, str(offset) + ".png")
            image_save_path = os.path.join(video, "colmap_" + str(offset), "input", folder + ".png")

            input_image_folder = os.path.join(video, "colmap_" + str(offset), "input")
            if not os.path.exists(input_image_folder):
                os.makedirs(input_image_folder)
            assert os.path.exists(image_path)
            image = cv2.imread(image_path).astype(np.float32)  # / 255.0
            h, w = image.shape[:2]

            image_size = (w, h)
            knew = np.zeros((3, 3), dtype=np.float32)

            knew[0, 0] = focal_scale * intrinsics[0, 0]
            knew[1, 1] = focal_scale * intrinsics[1, 1]
            knew[0, 2] = view["principal_point"][0]  # cx fixed half of the width
            knew[1, 2] = view["principal_point"][1]  #
            knew[2, 2] = 1.0

            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                intrinsics, dis_cef, R=None, P=knew, size=(w, h), m1type=cv2.CV_32FC1
            )

            undistorted_image = cv2.remap(
                image, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
            )
            undistorted_image = undistorted_image.clip(0, 255.0).astype(np.uint8)

            cv2.imwrite(image_save_path, undistorted_image)

            if offset == 0:
                distorting_flow = get_distorted_flow(
                    image, intrinsics, dis_cef, "linear", crop_output=False, scale=1.0, knew=knew
                )
                print("saved distortion mappers")
                np.save(os.path.join(video, folder + ".npy"), distorting_flow)


def soft_link_dataset(original_path, path, src_scene, scene):
    video_folder_list = glob.glob(original_path + "camera_*/")

    if not os.path.exists(path):
        os.makedirs(path)

    for video_folder in video_folder_list:
        new_link = os.path.join(path, video_folder.split("/")[-2])
        if os.path.exists(new_link):
            print("already exists do not make soft_link again")
            quit()
        assert not os.path.exists(new_link)
        cmd = " ln -s " + video_folder + " " + new_link
        os.system(cmd)
        print(cmd)

    original_model = original_path + "models.json"
    new_model = path + "models.json"
    shutil.copy(original_model, new_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", default="", type=str)
    parser.add_argument("--start_frame", "-s", default=0, type=int)
    parser.add_argument("--end_frame", "-e", default=50, type=int)

    args = parser.parse_args()
    video_path = args.video_path

    start_frame = args.start_frame
    end_frame = args.end_frame
    if start_frame >= end_frame:
        print("start frame must smaller than end frame")
        quit()
    if start_frame < 0 or end_frame > 300:
        print("frame must in range 0-300")
        quit()
    if not os.path.exists(video_path):
        print("path not exist")
        quit()

    if not video_path.endswith("/"):
        video_path = video_path + "/"

    src_scene = video_path.split("/")[-2]
    if src_scene not in immersive_seven:
        print("scene not in immersive_seven", immersive_seven)
        print("Please check if the scene name is correct")
        quit()

    if "04_Trucks" in video_path:
        print("04_Trucks")
        if end_frame > 150:
            end_frame = 150

    postfix = "_dist"  # distorted model

    scene = src_scene + postfix
    original_path = video_path  # "
    original_video = original_path  # 43 1
    path = video_path[:-1] + postfix
    video = original_path  # 43 1
    scale = immersive_scale_dict[scene]

    videos_list = glob.glob(original_video + "*.mp4")
    for v in tqdm.tqdm(videos_list):
        extract_frames(v)

    try:
        soft_link_dataset(original_path, path, src_scene, scene)
    except:
        print("soft_link failed")
        quit()
    try:
        image_undistort(
            video, offset_list=[i for i in range(start_frame, end_frame)], focal_scale=scale, fix_focal=None
        )
    except:
        print("undistort failed")
        quit()

    try:
        for offset in tqdm.tqdm(range(start_frame, end_frame)):
            convert_model_to_db_files(video, offset=offset, scale=scale, remove_everything_except_input=False)
    except:
        convert_model_to_db_files(video, offset=offset, scale=scale, remove_everything_except_input=True)
        print("create colmap input failed, better clean the data and try again")
        quit()
    for offset in range(start_frame, end_frame):
        get_colmap_single_im_distort(video, offset=offset)
