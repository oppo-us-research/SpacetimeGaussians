import os
import subprocess

from multiprocessing import Pool

import cv2
import numpy as np

from tqdm import tqdm


def cmd_wrapper(program):
    # print(program)
    os.system(program)


def images_to_video(img_folder, img_post_fix, output_vid_file, fps=25):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        f"/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps}",
        "-pattern_type",
        "glob",
        "-i",
        f"{img_folder}/*{img_post_fix}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-y",
        f"{output_vid_file}",
    ]

    # print(f'Running "{" ".join(command)}"')
    subprocess.call(command)


def create_white_images(img_folder):
    src_images = os.listdir(img_folder)
    src_images = [src_image for src_image in src_images if "_src" in src_image]
    for src_image in src_images:
        src_image_path = os.path.join(img_folder, src_image.replace("_src", "_white"))
        white_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        cv2.imwrite(src_image_path, white_img)


def video_to_images(vid_file, img_folder, img_post_fix, fps=25, verbose=False):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        f"/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        f"{vid_file}",
        "-vf",
        f"fps={fps}",
        "-qscale:v",
        "1",
        "-qmin",
        "1",
        "-qmax",
        "1",
        "-vsync",
        "0",
        f"{img_folder}/%06d{img_post_fix}",
    ]

    if verbose:
        print(f'Running "{" ".join(command)}"')
    subprocess.call(command)
