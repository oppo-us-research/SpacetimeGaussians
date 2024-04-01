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
import csv
import glob
import json
import os
import pickle
import shutil
import struct
import sys

import cv2
import natsort
import numpy as np
import tqdm

from PIL import Image

from thirdparty.colmap.pre_colmap import COLMAPDatabase
from thirdparty.gaussian_splatting.helper3dg import get_colmap_single_techni


def convert_model_to_db_files(path, offset=0):
    project_folder = os.path.join(path, "colmap_" + str(offset))
    manual_folder = os.path.join(project_folder, "manual")

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

    with open(os.path.join(path, "cameras_parameters.txt"), "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            idx = idx - 1
            row = [float(c) for c in row if c.strip() != ""]
            fx = row[0]
            fy = row[0]

            cx = row[1]
            cy = row[2]

            colmapQ = [row[5], row[6], row[7], row[8]]
            colmapT = [row[9], row[10], row[11]]
            camera_name = "cam" + str(idx).zfill(2)
            focal_length = fx

            principle_point = [0, 0]
            principle_point[0] = cx
            principle_point[1] = cy

            image_id = str(idx + 1)
            camera_id = image_id
            png_name = camera_name + ".png"

            line = image_id + " "

            for j in range(4):
                line += str(colmapQ[j]) + " "
            for j in range(3):
                line += str(colmapT[j]) + " "
            line = line + camera_id + " " + png_name + "\n"
            empty_line = "\n"
            image_txt_list.append(line)
            image_txt_list.append(empty_line)

            new_width = 2048
            new_height = 1088
            params = np.array(
                (
                    fx,
                    fy,
                    cx,
                    cy,
                )
            )

            camera_id = db.add_camera(1, new_width, new_height, params)
            # RADIAL_FISHEYE                                                                                 # width and height

            camera_line = (
                str(idx + 1)
                + " "
                + "PINHOLE "
                + str(new_width)
                + " "
                + str(new_height)
                + " "
                + str(focal_length)
                + " "
                + str(focal_length)
                + " "
                + str(cx)
                + " "
                + str(cy)
                + "\n"
            )
            camera_txt_list.append(camera_line)
            image_id = db.add_image(
                png_name,
                camera_id,
                prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])),
                prior_t=np.array((colmapT[0], colmapT[1], colmapT[2])),
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


def image_copy(video, offset_list=[0], focal_scale=1.0, fix_focal=None):

    png_list = glob.glob(video + "/*.png")

    for png_path in png_list:
        pass

    for idx, offset in enumerate(offset_list):
        png_list = glob.glob(video + "*_undist_" + str(offset).zfill(5) + "_*.png")

        target_folder = os.path.join(video, "colmap_" + str(idx), "input")
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        for png_path in png_list:
            camera_name = os.path.basename(png_path).split("_")[3]
            new_path = os.path.join(target_folder, "cam" + camera_name)
            shutil.copy(png_path, new_path)


def check_image(video_path):
    import cv2

    from PIL import Image

    image_list = glob.glob(video_path + "*.png")
    for image_path in image_list:
        try:
            img = Image.open(image_path)  # open the image file
            img.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print("Bad file:", image_path)  # print out the names of corrupt files
        bad_file_list = []
        bad_count = 0
        try:
            img.cv2.imread(image_path)
            shape = img.shape  # this will throw an error if the img is not read correctly
        except:
            bad_file_list.append(image_path)
            bad_count += 1
    print(bad_file_list)


def fix_broken(image_path, ref_image_path):
    try:
        img = Image.open(image_path)  # open the image file
        print("start verifying", image_path)
        img.verify()  # if we already fixed it.
        print("already fixed", image_path)
    except:
        print("Bad file:", image_path)
        import cv2

        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(image_path)

        img.load()
        img.save("tmp.png")

        saved_image = cv2.imread("tmp.png")
        mask = saved_image == 0
        ref_image = cv2.imread(ref_image_path)
        composed = saved_image * (1 - mask) + ref_image * (mask)
        cv2.imwrite(image_path, composed)
        print("fixing done", image_path)
        os.remove("tmp.png")


if __name__ == "__main__":
    scene_name_list = ["Train"]
    frame_range_dict = {}
    frame_range_dict["Birthday"] = [_ for _ in range(151, 201)]  # start from 1
    frame_range_dict["Fabien"] = [_ for _ in range(51, 101)]  # start from 1
    frame_range_dict["Painter"] = [_ for _ in range(100, 150)]  # start from 0
    frame_range_dict["Theater"] = [_ for _ in range(51, 101)]  # start from 1
    frame_range_dict["Train"] = [_ for _ in range(151, 201)]  # start from 1

    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", default="", type=str)
    args = parser.parse_args()

    video_path = args.video_path

    if not video_path.endswith("/"):
        video_path = video_path + "/"

    src_scene = video_path.split("/")[-2]
    print("src_scene", src_scene)

    if src_scene == "Birthday":
        print("check broken")
        fix_broken(video_path + "Birthday_undist_00173_09.png", video_path + "Birthday_undist_00172_09.png")

    image_copy(video_path, offset_list=frame_range_dict[src_scene])
    # #
    for offset in tqdm.tqdm(range(0, 50)):
        convert_model_to_db_files(video_path, offset=offset)

    for offset in range(0, 50):
        get_colmap_single_techni(video_path, offset=offset)

    #  rm -r colmap_* # once meet error, delete all colmap_* folders and rerun this script.
