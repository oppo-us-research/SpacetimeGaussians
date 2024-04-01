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
import sys

import cv2
import numpy as np

from tqdm import tqdm, trange

from thirdparty.colmap.pre_colmap import COLMAPDatabase
from thirdparty.gaussian_splatting.helper3dg import get_colmap_single_n3d
from thirdparty.gaussian_splatting.utils.my_utils import (
    pose_to_w2c_matrixes,
    rot_mat_2_qvec,
)


def extract_frames(video_path):
    cam = cv2.VideoCapture(video_path)
    video_true_frame_num = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    ctr = 0
    success = True
    for i in range(video_true_frame_num):
        if os.path.exists(os.path.join(video_path.replace(".mp4", ""), str(i) + ".png")):
            ctr += 1
    if ctr == video_true_frame_num:  # or ctr == 150:  # 150 for 04_truck
        print("already extracted all the frames, skip extracting")
        return
    ctr = 0
    while ctr < video_true_frame_num:
        try:
            _, frame = cam.read()

            save_path = os.path.join(video_path.replace(".mp4", ""), str(ctr) + ".png")
            if not os.path.exists(video_path.replace(".mp4", "")):
                os.makedirs(video_path.replace(".mp4", ""))

            cv2.imwrite(save_path, frame)
            ctr += 1
        except:
            break
            success = False
            # print("error")
    print(f"extracted {ctr} frames from {video_path}")
    cam.release()
    return


def prepare_colmap_dynerf(folder, offset=0):
    folder_list = glob.glob(folder + "cam**/")
    save_dir = os.path.join(folder, "colmap_" + str(offset))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, "input")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for folder in folder_list:
        image_path = os.path.join(folder, str(offset) + ".png")
        image_save_path = os.path.join(save_dir, folder.split("/")[-2] + ".png")

        shutil.copy(image_path, image_save_path)


def convert_dynerf_to_colmap_db(path, offset=0):
    origin_numpy = os.path.join(path, "poses_bounds.npy")
    video_paths = sorted(glob.glob(os.path.join(path, "cam*.mp4")))
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

    with open(origin_numpy, "rb") as numpy_file:
        poses_bounds = np.load(numpy_file)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)

        llff_poses = poses.copy().transpose(1, 2, 0)
        w2c_matrixes_list = pose_to_w2c_matrixes(llff_poses)
        assert type(w2c_matrixes_list) == list

        for i in range(len(poses)):
            camera_name = os.path.basename(video_paths[i])[:-4]  # "cam" + str(i).zfill(2)
            m = w2c_matrixes_list[i]
            colmapR = m[:3, :3]
            T = m[:3, 3]

            H, W, focal = poses[i, :, -1]

            colmapQ = rot_mat_2_qvec(colmapR)
            # colmapR_check = qvec_2_rot_mat(colmapQ)

            image_id = str(i + 1)
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

            focal_length = focal
            model, width, height, params = (
                i,
                W,
                H,
                np.array(
                    (
                        focal_length,
                        focal_length,
                        W // 2,
                        H // 2,
                    )
                ),
            )

            camera_id = db.add_camera(1, width, height, params)
            camera_line = (
                str(i + 1)
                + " "
                + "PINHOLE "
                + str(width)
                + " "
                + str(height)
                + " "
                + str(focal_length)
                + " "
                + str(focal_length)
                + " "
                + str(W // 2)
                + " "
                + str(H // 2)
                + "\n"
            )
            camera_txt_list.append(camera_line)

            image_id = db.add_image(
                png_name,
                camera_id,
                prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])),
                prior_t=np.array((T[0], T[1], T[2])),
                image_id=i + 1,
            )
            db.commit()
        db.close()

    with open(save_txt, "w") as f:
        for line in image_txt_list:
            f.write(line)
    with open(save_camera, "w") as f:
        for line in camera_txt_list:
            f.write(line)
    with open(save_points, "w") as f:
        pass


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

    ## step 1
    print(f"start extracting {end_frame - start_frame} frames from videos")
    videos_list = glob.glob(video_path + "*.mp4")
    for v in tqdm(videos_list):
        extract_frames(v)

    ### step 2 prepare colmap input
    print("start preparing colmap image input")
    for frame_idx_offset in trange(start_frame, end_frame):
        prepare_colmap_dynerf(video_path, frame_idx_offset)

    print("start preparing colmap database input")
    ### step 3 prepare colmap db file
    for frame_idx_offset in trange(start_frame, end_frame):
        convert_dynerf_to_colmap_db(video_path, frame_idx_offset)

    ### step 4 run colmap, per frame, if error, reinstall opencv-headless
    for frame_idx_offset in trange(start_frame, end_frame):
        get_colmap_single_n3d(video_path, frame_idx_offset)
