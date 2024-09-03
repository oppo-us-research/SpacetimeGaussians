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

import os
from pathlib import Path

import cv2 
import glob 
import tqdm 
import numpy as np 
import shutil
import pickle
import sys 
import argparse


sys.path.append(".")
from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec
from thirdparty.colmap.pre_colmap import * 
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglen3d
from script.utils_pre import write_colmap




def extractframes(videopath: Path, startframe=0, endframe=300, downscale=1, save_subdir = '', ext='png'):
    output_dir = videopath.parent / save_subdir / videopath.stem
        
    if all((output_dir / f"{i}.{ext}").exists() for i in range(startframe, endframe)):
        print(f"Already extracted all the frames in {output_dir}")
        return

    cam = cv2.VideoCapture(str(videopath))
    cam.set(cv2.CAP_PROP_POS_FRAMES, startframe)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(startframe, endframe):
        success, frame = cam.read()
        if not success:
            print(f"Error reading frame {i}")
            break

        if downscale > 1:
            new_width, new_height = int(frame.shape[1] / downscale), int(frame.shape[0] / downscale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(output_dir / f"{i}.{ext}"), frame)

    cam.release()


def preparecolmapdynerf(folder, offset=0):
    folderlist = sorted(folder.glob("cam??/"))

    savedir = folder / f"colmap_{offset}" / "input"
    savedir.mkdir(exist_ok=True, parents=True)

    for folder in folderlist:
        imagepath = folder / f"{offset}.png"
        imagesavepath = savedir / f"{folder.name}.png"

        if (imagesavepath.exists()):
            continue

        assert imagepath.exists
        # shutil.copy(imagepath, imagesavepath)
        imagesavepath.symlink_to(imagepath.resolve())


def convertdynerftocolmapdb(path, offset=0, downscale=1):
    originnumpy = path / "poses_bounds.npy"
    video_paths = sorted(path.glob('cam*.mp4'))

    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)

        llffposes = poses.copy().transpose(1, 2, 0)
        w2c_matriclist = posetow2c_matrcs(llffposes)
        assert (type(w2c_matriclist) == list)

        cameras = []
        for i in range(len(poses)):
            cameraname = video_paths[i].stem
            m = w2c_matriclist[i]
            colmapR = m[:3, :3]
            T = m[:3, 3]

            H, W, focal = poses[i, :, -1] / downscale

            colmapQ = rotmat2qvec(colmapR)

            camera = {
                'id': i + 1,
                'filename': f"{cameraname}.png",
                'w': W,
                'h': H,
                'fx': focal,
                'fy': focal,
                'cx': W // 2,
                'cy': H // 2,
                'q': colmapQ,
                't': T,
            }
            cameras.append(camera)

    write_colmap(path, cameras, offset)



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=50, type=int)
    parser.add_argument("--downscale", default=1, type=int)

    args = parser.parse_args()
    videopath = Path(args.videopath)

    startframe = args.startframe
    endframe = args.endframe
    downscale = args.downscale

    print(f"params: startframe={startframe} - endframe={endframe} - downscale={downscale} - videopath={videopath}")

    if startframe >= endframe:
        print("start frame must smaller than end frame")
        quit()
    if startframe < 0 or endframe > 300:
        print("frame must in range 0-300")
        quit()
    if not videopath.exists():
        print("path not exist")
        quit()
    ##### step1
    print("start extracting 300 frames from videos")
    videoslist = sorted(videopath.glob("*.mp4"))
    for v in tqdm.tqdm(videoslist, desc="Extract frames from videos"):
        extractframes(v, downscale=downscale)

    

    # # ## step2 prepare colmap input 
    print("start preparing colmap image input")
    for offset in range(startframe, endframe):
        preparecolmapdynerf(videopath, offset)


    print("start preparing colmap database input")
    # # ## step 3 prepare colmap db file 
    for offset in tqdm.tqdm(range(startframe, endframe), desc="convertdynerftocolmapdb"):
        convertdynerftocolmapdb(videopath, offset, downscale)


    # ## step 4 run colmap, per frame, if error, reinstall opencv-headless 
    for offset in range(startframe, endframe):
        getcolmapsinglen3d(videopath, offset)

