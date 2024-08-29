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




def extractframes(videopath: Path, startframe=0, endframe=300, downscale=1):
    output_dir = videopath.with_suffix('')
    if all((output_dir / f"{i}.png").exists() for i in range(startframe, endframe)):
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

        cv2.imwrite(str(output_dir / f"{i}.png"), frame)

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
    projectfolder = path / f"colmap_{offset}"
    manualfolder = projectfolder / "manual"

    manualfolder.mkdir(exist_ok=True)

    savetxt = manualfolder / "images.txt"
    savecamera = manualfolder / "cameras.txt"
    savepoints = manualfolder / "points3D.txt"

    imagetxtlist = []
    cameratxtlist = []

    db_path = projectfolder / "input.db"
    if db_path.exists():
        db_path.unlink()

    db = COLMAPDatabase.connect(db_path)
    db.create_tables()

    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)

        llffposes = poses.copy().transpose(1, 2, 0)
        w2c_matriclist = posetow2c_matrcs(llffposes)
        assert (type(w2c_matriclist) == list)

        for i in range(len(poses)):
            cameraname = video_paths[i].stem
            m = w2c_matriclist[i]
            colmapR = m[:3, :3]
            T = m[:3, 3]

            H, W, focal = poses[i, :, -1] / downscale

            colmapQ = rotmat2qvec(colmapR)
            # colmapRcheck = qvec2rotmat(colmapQ)

            imageid = str(i + 1)
            cameraid = imageid
            pngname = f"{cameraname}.png"

            line = f"{imageid} " + " ".join(map(str, colmapQ)) + " " + " ".join(
                map(str, T)) + f" {cameraid} {pngname}\n"
            imagetxtlist.append(line)
            imagetxtlist.append("\n")

            focolength = focal
            model, width, height, params = i, W, H, np.array((focolength, focolength, W // 2, H // 2,))

            camera_id = db.add_camera(1, width, height, params)
            cameraline = f"{i + 1} PINHOLE {width} {height} {focolength} {focolength} {W // 2} {H // 2}\n"
            cameratxtlist.append(cameraline)

            image_id = db.add_image(pngname, camera_id,
                                    prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])),
                                    prior_t=np.array((T[0], T[1], T[2])), image_id=i + 1)
            db.commit()
        db.close()

    savetxt.write_text("".join(imagetxtlist))
    savecamera.write_text("".join(cameratxtlist))
    savepoints.write_text("")  # Creating an empty points3D.txt file


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

