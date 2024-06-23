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
from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec
from thirdparty.colmap.pre_colmap import * 
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglen3d
from thirdparty.gaussian_splatting.colmap_loader import read_extrinsics_binary, read_intrinsics_binary


def get_cam_name(video_path):
    return os.path.splitext(os.path.basename(video_path))[0] 

def extract_frames(video_path, start_frame_num, end_frame_num, image_ext) :
    """
    Extracts a specified number of frames from a video and saves them as PNG files.
    
    Args:
    video_path (str): The path to the video file.
    duration (int): The number of frames to extract.
    """
    cam = cv2.VideoCapture(video_path)
    video_dir = os.path.dirname(video_path)
    save_dir = os.path.join(video_dir, "frames")
    video_counter = 0
    success = True
    # video name
    video_name = get_cam_name(video_path)
    # output dir 
    output_dir = os.path.join(save_dir, video_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    while video_counter < end_frame_num:
        try:
            ret, frame = cam.read()
            if not ret:
                print(f"Failed to read frame at {video_counter}. Ending extraction.")
                break
            save_path = os.path.join(output_dir, f"{video_counter}."+image_ext)
            if os.path.exists(save_path):
                    print("skiped")
            if video_counter > start_frame_num -1:

                cv2.imwrite(save_path, frame)
            video_counter += 1 
        except Exception as e:
            success = False
            print(f"Error while extracting frames: {e}")
            break
    
    cam.release()
    if success:
        print(f"Successfully extracted {video_counter} frames from {video_path}")
    else:
        print(f"Failed to extract frames from {video_path}")


def prepare_colmap(folder, offset, extension, point_root):
    folderlist =  [os.path.join(folder, sub_dir) for sub_dir in os.listdir(folder) ] 


    savedir = os.path.join(point_root, "colmap_" + str(offset))
    savedir = os.path.join(savedir, "input")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for folder in folderlist :
        imagepath = os.path.join(folder, str(offset) + "." + extension)
        imagesavepath = os.path.join(savedir, folder.split("/")[-1] + "." + extension)

        shutil.copy(imagepath, imagesavepath)


    


    
def convert_selected_cam_matrix_to_colmapdb(path, offset=0,ref_frame=0,image_ext="png"):
    
    # 

    projectfolder = os.path.join(path, "colmap_" + str(offset))
    refprojectfolder =  os.path.join(path, "colmap_" + str(ref_frame))
    manualfolder = os.path.join(projectfolder, "manual")

    
    cameras_extrinsic_file = os.path.join(refprojectfolder, "distorted/sparse/0/", "images.bin") # from distorted?
    cameras_intrinsic_file = os.path.join(refprojectfolder, "distorted/sparse/0/", "cameras.bin")

    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    if not os.path.exists(manualfolder):
        os.makedirs(manualfolder)
     
    
    savetxt = os.path.join(manualfolder, "images.txt")
    savecamera = os.path.join(manualfolder, "cameras.txt")
    savepoints = os.path.join(manualfolder, "points3D.txt")
    imagetxtlist = []
    cameratxtlist = []
    if os.path.exists(os.path.join(projectfolder, "input.db")):
        print("remove previus db file")
        os.remove(os.path.join(projectfolder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(projectfolder, "input.db"))

    db.create_tables()


    helperdict = {}
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)
        helperdict[extr.name] = [extr, intr]
    
    sortedtotalcamelist =  natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in  range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i # map each cam with a number

    videopath = glob.glob(refprojectfolder + "/images/*."+image_ext)
    for i in range(len(videopath)):
        cameraname = get_cam_name(videopath[i])  #"cam" + str(i).zfill(2)
        cameranameaskey = cameraname + "." + image_ext
        extr, intr =  helperdict[cameranameaskey] # extr.name

        width, height, params = intr.width, intr.height, intr.params
        focolength = intr.params[0]
        fw, fh = intr.params[2], intr.params[3]

        colmapQ = extr.qvec
        T = extr.tvec
        

        imageid = str(i+1)
        cameraid = imageid
        pngname = cameraname + "." + image_ext
        
        line =  imageid + " "

        for j in range(4):
            line += str(colmapQ[j]) + " "
        for j in range(3):
            line += str(T[j]) + " "
        line = line  + cameraid + " " + pngname + "\n"
        empltyline = "\n"
        imagetxtlist.append(line)
        imagetxtlist.append(empltyline)

        


        camera_id = db.add_camera(4, width, height, params)
        strparams = [str(ele) for ele in params]
        cameraline = str(i+1) + " " + "OPENCV " + str(width) +  " " + str(height)  +  " " + " ".join(strparams) + " \n"
        cameratxtlist.append(cameraline)
        
        image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((T[0], T[1], T[2])), image_id=i+1)
        db.commit()
    db.close()

    with open(savetxt, "w") as f:
        for line in imagetxtlist :
            f.write(line)
    with open(savecamera, "w") as f:
        for line in cameratxtlist :
            f.write(line)
    with open(savepoints, "w") as f:
        pass 
    print("done")

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
    videos_dir = args.videosdir
    start_frame_num = args.startframe
    end_frame_num = args.endframe
    duration = args.endframe - args.startframe
    
    # archor frame number offset to get pose.
    pose_ref_frame_num = args.refframe
  
    

    # input checking
    if start_frame_num >= end_frame_num:
        print("start frame must smaller than end frame")
        quit()

    if not os.path.exists(videos_dir):
        print("path not exist")
        quit()
    
    if not videos_dir.endswith("/"):
        videos_dir = videos_dir + "/"
    
    #step 1 videos to pngs. TODO .jpg but jpg contains artifacts.
    print(f"Start extracting {duration} frames from videos at {videos_dir}")
    video_path_list = glob.glob(os.path.join(videos_dir, "*.mp4"))
    for video_path in tqdm.tqdm(video_path_list):
        extract_frames(video_path, start_frame_num, end_frame_num, image_ext)
    
    
    # create video path
    decoded_frame_root = os.path.join(videos_dir, "frames")

    ## step2 prepare colmap input 
    point_root = os.path.join(videos_dir, "point")
    print("start preparing colmap image input")
    for offset in range(start_frame_num, end_frame_num):
        prepare_colmap(decoded_frame_root, offset, image_ext, point_root)



    # step3, colmap without gt pose 
    colmap_project_root = os.path.join(videos_dir, "point")

    pose_ref_frame_project = os.path.join(colmap_project_root, "colmap_" + str(pose_ref_frame_num))
    cmd = "python thirdparty/gaussian_splatting/convert.py -s " + pose_ref_frame_project
    
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
