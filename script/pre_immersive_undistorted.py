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

from thirdparty.colmap.pre_colmap import *
from thirdparty.gaussian_splatting.helper3dg import get_colmap_single_im_undistort
from thirdparty.gaussian_splatting.utils.my_utils import rot_mat_2_qvec


SCALE_DICT = {}

# SCALE_DICT["01_Welder_S11"] = 0.35
# ["04_Truck", "09_Alexa", "10_Alexa", "11_Alexa", "12_Cave"]
immersive_seven = ["01_Welder", "02_Flames", "04_Truck", "09_Alexa", "10_Alexa", "11_Alexa", "12_Cave"]
immersive_scale_dict = {}
immersive_scale_dict["01_Welder"] = 1.0
immersive_scale_dict["02_Flames"] = 1.0
immersive_scale_dict["04_Truck"] = 1.0
immersive_scale_dict["09_Alexa"] = 1.0
immersive_scale_dict["10_Alexa"] = 1.0
immersive_scale_dict["11_Alexa"] = 1.0
immersive_scale_dict["12_Cave"] = 1.0

for scene in immersive_seven:
    SCALE_DICT[scene + "_dist"] = 1.0  #
    immersive_scale_dict[scene + "_dist"] = 1.0


def extract_frames(video_path):
    cam = cv2.VideoCapture(video_path)
    ctr = 0
    while ctr < 300:
        _, frame = cam.read()

        save_path = os.path.join(video_path.replace(".mp4", ""), str(ctr) + ".png")
        if not os.path.exists(video_path.replace(".mp4", "")):
            os.makedirs(video_path.replace(".mp4", ""))
        cv2.imwrite(save_path, frame)
        ctr += 1
    cam.release()
    return


def extract_frames_x1(video_path):
    cam = cv2.VideoCapture(video_path)
    ctr = 0
    success = True
    while ctr < 300:
        try:
            _, frame = cam.read()

            save_path = os.path.join(video_path.replace(".mp4", ""), str(ctr) + ".png")
            if not os.path.exists(video_path.replace(".mp4", "")):
                os.makedirs(video_path.replace(".mp4", ""))

            cv2.imwrite(save_path, frame)
            ctr += 1
        except:
            success = False
            cam.release()
            return

    cam.release()
    return


def convert_model_to_db_files(path, offset=0, scale=1.0):
    project_folder = os.path.join(path, "colmap_" + str(offset))
    # sparse_folder = os.path.join(project_folder, "sparse/0")
    manual_folder = os.path.join(project_folder, "manual")

    # if not os.path.exists(sparse_folder):
    #     os.makedirs(sparse_folder)
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
        Knew[0, 2] = view["principal_point"][0]
        Knew[1, 2] = view["principal_point"][1]

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

        camera_id = db.add_camera(
            1, new_width, new_height, params
        )  # RADIAL_FISHEYE                                                                                 # width and height
        #

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
    parser.add_argument("--start_frame", default=0, type=int)
    parser.add_argument("--end_frame", default=50, type=int)

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

    postfix = "_undist"  # undistorted cameras

    scene = src_scene + postfix
    original_path = video_path  #
    original_video = original_path  # 43 1
    path = video_path[:-1] + postfix
    video = original_path  # 43 1
    scale = immersive_scale_dict[scene]

    videos_list = glob.glob(original_video + "*.mp4")
    for v in tqdm.tqdm(videos_list):
        extract_frames_x1(v)

    soft_link_dataset(original_path, path, src_scene, scene)

    try:
        image_undistort(
            video, offset_list=[i for i in range(start_frame, end_frame)], focal_scale=scale, fix_focal=None
        )
    except:
        print("undistort failed")
        quit()

    try:
        for offset in tqdm.tqdm(range(start_frame, end_frame)):
            convert_model_to_db_files(video, offset=offset, scale=scale)
    except:
        print("create colmap input failed, better clean the data and try again")
        quit()

    for offset in range(0, 50):
        get_colmap_single_im_undistort(video, offset=offset)
