# MIT License

# Copyright (c) 2024 OPPO

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
import natsort
sys.path.append(".")

from script.pre_n3d import extractframes
from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec
from thirdparty.colmap.pre_colmap import * 
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglen3d
from thirdparty.gaussian_splatting.colmap_loader import read_extrinsics_binary, read_intrinsics_binary


def get_cam_name(video_path):
    return os.path.splitext(os.path.basename(video_path))[0] 


def prepare_colmap(folder, offset, extension, point_root):
    folderlist =  sorted(folder.iterdir())

    savedir = point_root / f"colmap_{offset}" / "input"
    savedir.mkdir(exist_ok=True, parents=True)
        
    for folder in folderlist :
        imagepath = folder / f"{offset}.{extension}"
        imagesavepath = savedir / f"{folder.name}.{extension}"
        
        if (imagesavepath.exists()):
            continue
            
        imagesavepath.symlink_to(imagepath.resolve())

    


    
def convert_selected_cam_matrix_to_colmapdb(path, offset=0,ref_frame=0,image_ext="png"):
    
    # 

    projectfolder = path / f"colmap_{offset}"
    refprojectfolder =  path / f"colmap_{ref_frame}"
    manualfolder = projectfolder / "manual"

    
    cameras_extrinsic_file = refprojectfolder / "distorted" / "sparse" / "0" / "images.bin" # from distorted?
    cameras_intrinsic_file = refprojectfolder / "distorted" / "sparse" / "0" / "cameras.bin"

    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    manualfolder.mkdir(exist_ok=True)     
    
    savetxt = manualfolder / "images.txt"
    savecamera = manualfolder / "cameras.txt"
    savepoints = manualfolder / "points3D.txt"
    imagetxtlist = []
    cameratxtlist = []

    db_file = projectfolder / "input.db"
    if db_file.exists():
        db_file.unlink()

    db = COLMAPDatabase.connect(db_file)

    db.create_tables()
        
    cam_extrinsics_by_name = {extr.name: extr for extr in cam_extrinsics.values()}
    
    videopaths = sorted((refprojectfolder / "images").glob(f"*.{image_ext}"))
    
    for i, videopath in enumerate(videopaths):
        filename = videopath.name #eg cam00.png
        extr = cam_extrinsics_by_name[filename]
        intr = cam_intrinsics[extr.camera_id]

        w, h, params = intr.width, intr.height, intr.params
        focolength = intr.params[0]
        fx, fy = intr.params[2], intr.params[3]

        colmapQ = extr.qvec
        T = extr.tvec
        

        id = str(i+1)
        
        line = f"{id} " + " ".join(map(str, colmapQ)) + " " + " ".join(map(str, T)) + f" {id} {filename}\n"
        imagetxtlist.append(line)
        imagetxtlist.append("\n")

        camera_id = db.add_camera(4, w, h, params)
        cameraline = f"{id} OPENCV {w} {h} {' '.join(params.astype(str))} \n"
        cameratxtlist.append(cameraline)
        
        image_id = db.add_image(filename, camera_id,  prior_q=colmapQ, prior_t=T, image_id=id)
        db.commit()
    db.close()

    savetxt.write_text("".join(imagetxtlist))
    savecamera.write_text("".join(cameratxtlist))
    savepoints.write_text("")

if __name__ == "__main__" :
    """
    give a videos in 
    videosroot/o.mp4
    videosroot/1.mp4
    TODO support fisheye videos with given prior.

    1. extract video to frames:   "videosroot/frame/"
    2. move frames to each point colmap project folder:   "videosroot/point/colmap_*/"
    3. use ref frame to get camera model (opencv) and get ref frame's point: "videosroot/point/colmap_0/"  to "videosroot/point/colmap_*/"
    4. apply opencv camera model to all the other frames
    5. get SfM points for all the other frames 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--imageext", default="png", type=str)

    parser.add_argument("--videosdir", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=50, type=int)
    parser.add_argument("--refframe", default=0, type=int)

    args = parser.parse_args()

    # check image extension
    if args.imageext not in ["png","jpeg", "jpg"]:
        print("wrong extension")
        quit()
   
    # get input config
    image_ext = args.imageext
    videos_dir = Path(args.videosdir)
    start_frame_num = args.startframe
    end_frame_num = args.endframe
    duration = args.endframe - args.startframe
    
    # archor frame number offset to get pose.
    pose_ref_frame_num = args.refframe
  
    

    # input checking
    if start_frame_num >= end_frame_num:
        print("start frame must smaller than end frame")
        quit()

    if not videos_dir.exists():
        print("path not exist")
        quit()

    
    #step 1 videos to pngs. TODO .jpg but jpg contains artifacts.
    print(f"Start extracting {duration} frames from videos at {videos_dir}")
    video_path_list = sorted(videos_dir.glob("*.mp4"))
    for video_path in tqdm.tqdm(video_path_list):
        extractframes(video_path, start_frame_num, end_frame_num, save_subdir='frames', ext=image_ext)
    
    
    # create video path
    decoded_frame_root = videos_dir / "frames"

    ## step2 prepare colmap input 
    point_root = videos_dir / "point"
    print("start preparing colmap image input")
    for offset in range(start_frame_num, end_frame_num):
        prepare_colmap(decoded_frame_root, offset, image_ext, point_root)



    # step3, colmap without gt pose 
    colmap_project_root = videos_dir / "point"

    pose_ref_frame_project = colmap_project_root / f"colmap_{pose_ref_frame_num}"
    cmd = f"python thirdparty/gaussian_splatting/convert.py -s {pose_ref_frame_project}"

    exit_code = os.system(cmd)
    if exit_code != 0:
        exit(exit_code)

    # step4, use that pose/intrics for the rest models 
    for frame_num in range(start_frame_num, end_frame_num):
        if frame_num != pose_ref_frame_num:
            convert_selected_cam_matrix_to_colmapdb(colmap_project_root, frame_num, pose_ref_frame_num, image_ext)



    for frame_num in range(start_frame_num, end_frame_num):
        if frame_num != pose_ref_frame_num:
            getcolmapsinglen3d(colmap_project_root, frame_num)
