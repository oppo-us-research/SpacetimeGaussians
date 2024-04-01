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

import glob
import os

import cv2
import natsort
import numpy as np


def compare_spatial_temporal():
    method_dict = {}
    method_dict["ours"] = ""
    method_dict["ourslite"] = ""
    method_dict["dynamic3DGs"] = ""
    method_dict["hpreal"] = ""
    method_dict["kplane"] = ""
    method_dict["mixvoxel"] = ""
    method_dict["nerfplayer"] = ""

    method_dict["gt"] = method_dict["ours"].replace("/renders/", "/gt/")

    assert method_dict["ours"] != method_dict["gt"]

    top_left = (0, 0)  #
    y, x = top_left[0], top_left[1]

    delta_y = 150
    delta_x_frames = 250

    for k in method_dict.keys():
        total = []
        path = method_dict[k]
        if path != None:
            image_list = glob.glob(path)
            image_list = natsort.natsorted(image_list)
            print(k, len(image_list))
            image_list = image_list[50:]
            print(image_list[0])
            for image_path in image_list[0:delta_x_frames]:
                image = cv2.imread(image_path)
                patch = image[y : y + delta_y, x : x + 1, :]
                total.append(patch)
            final = np.hstack(total)
            cv2.imwrite("output" + str(k) + ".png", final)


def convert_videos():
    save_dir = "/home/output"
    path = "/renders/*.png"

    images = natsort.natsorted(glob.glob(path))

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    file_name = os.path.join(save_dir, "flame_steak.mp4")
    video = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))  # change fps by yourself

    for image_path in images:
        image = cv2.imread(image_path)
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


# code to look for matched image
def look_for_image():
    source_image = "xxx.jpg"
    image = cv2.imread(source_image)
    image_list = glob.glob("/home/gt/*.png")

    image_list = natsort.natsorted(image_list)
    image = cv2.resize(image.astype(np.float32), (1352, 1014), interpolation=cv2.INTER_CUBIC)
    max_psnr = 0
    max_psnr_path = 0
    second_path = 0
    second_psnr = 0
    for image_path in image_list:
        img2 = cv2.imread(image_path).astype(np.float32)
        img2 = cv2.resize(img2.astype(np.float32), (1352, 1014), interpolation=cv2.INTER_CUBIC)

        psnr = cv2.PSNR(image, img2)
        if psnr > max_psnr:
            second_psnr = max_psnr

            max_psnr = psnr
            second_path = max_psnr_path
            max_psnr_path = image_path

    print(max_psnr, max_psnr_path)
    print(second_psnr, second_path)


def remove_nfs():
    # remove not used nfs files
    nfs_list = glob.glob("/.nfs*")

    for f in nfs_list:
        cmd = " lsof -t " + f + " | xargs kill -9 "
        ret = os.system(cmd)
        print(cmd)


if __name__ == "__main__":
    # compare_spatial_temporal()
    # convert_videos()
    pass  #
