# MIT License
import json
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
import natsort 
import sys 
import struct
import pickle
from scipy.spatial.transform import Rotation
sys.path.append(".")
from thirdparty.gaussian_splatting.utils.my_utils import rotmat2qvec
from thirdparty.colmap.pre_colmap import * 
from thirdparty.gaussian_splatting.helper3dg import getcolmapsingleimundistort
from script.pre_n3d import extractframes
from script.pre_immersive_distorted import softlinkdataset, convertmodel2dbfiles

import argparse

SCALEDICT = {}


Immersiveseven = ["01_Welder",  "02_Flames", "04_Truck", "09_Alexa", "10_Alexa", "11_Alexa", "12_Cave"]
immmersivescaledict = {}
immmersivescaledict["01_Welder"] = 1.0
immmersivescaledict["02_Flames"] =1.0
immmersivescaledict["04_Truck"] = 1.0
immmersivescaledict["09_Alexa"] = 1.0
immmersivescaledict["10_Alexa"] = 1.0
immmersivescaledict["11_Alexa"] = 1.0
immmersivescaledict["12_Cave"] = 1.0

for scene in Immersiveseven:
    SCALEDICT[scene + "_undist"] = 0.5  # 
    immmersivescaledict[scene + "_undist"] = 0.5


def imageundistort_no_mapper(video, offsetlist=[0],focalscale=1.0, fixfocal=None):
    with open(video / "models.json", "r") as f:
        meta = json.load(f)

    for idx , camera in enumerate(tqdm.tqdm(meta, desc="undistort")):
        folder = camera['name'] # camera_0001
        view = camera
        intrinsics = np.array([[view['focal_length'], 0.0, view['principal_point'][0]],
                            [0.0, view['focal_length'], view['principal_point'][1]],
                            [0.0, 0.0, 1.0]])
        dis_cef = np.zeros((4))

        dis_cef[:2] = np.array(view['radial_distortion'])[:2]

        map1, map2 = None, None
        for offset in offsetlist:
            imagepath = video / folder / f"{offset}.png"
            imagesavepath = video / f"colmap_{offset}" / "input" / f"{folder}.png"

            inputimagefolder = video / f"colmap_{offset}" / "input"
            inputimagefolder.mkdir(exist_ok=True, parents=True)
            assert imagepath.exists()
            
            if not imagesavepath.exists():
                try:
                    image = cv2.imread(str(imagepath)).astype(np.float32) #/ 255.0
                except:
                    print("failed to read image", imagepath)
                    quit()
                    
                h, w = image.shape[:2]

                image_size = (w, h)
                knew = np.zeros((3, 3), dtype=np.float32)

    
                knew[0,0] = focalscale * intrinsics[0,0]
                knew[1,1] = focalscale * intrinsics[1,1]
                knew[0,2] =  view['principal_point'][0] # cx fixed half of the width
                knew[1,2] =  view['principal_point'][1] #
                knew[2,2] =  1.0


                
                
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(intrinsics, dis_cef, R=None, P=knew, size=(w, h), m1type=cv2.CV_32FC1)

                undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
                undistorted_image = undistorted_image.clip(0,255.0).astype(np.uint8)

                cv2.imwrite(imagesavepath, undistorted_image)
            else:
                print("already exists")







if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=50, type=int)


    args = parser.parse_args()
    videopath = Path(args.videopath)

    startframe = args.startframe
    endframe = args.endframe
    if startframe >= endframe:
        print("start frame must smaller than end frame")
        quit()
    if startframe < 0 or endframe > 300:
        print("frame must in range 0-300")
        quit()
    if not videopath.exists():
        print("path not exist")
        quit()

    srcscene = videopath.name
    if srcscene not in Immersiveseven:
        print("scene not in Immersiveseven", Immersiveseven)
        print("Please check if the scene name is correct")
        quit()
    

    if "04_Trucks" == srcscene:
        print('04_Trucks')
        if endframe > 150:
            endframe = 150 

    postfix  = "_undist" # undistored cameras

    scene = srcscene + postfix
    dstpath = videopath.with_name(videopath.name + postfix)
    scale = immmersivescaledict[scene]

    videoslist = sorted(videopath.glob("*.mp4"))
    for v in tqdm.tqdm(videoslist, desc="extract frames"):
        extractframes(v)


    softlinkdataset(videopath, dstpath)

    imageundistort_no_mapper(dstpath, offsetlist=list(range(startframe,endframe)),focalscale=scale, fixfocal=None)
  
    
    try:
        for offset in tqdm.tqdm(range(startframe, endframe), desc='convertmodel2dbfiles'):
            convertmodel2dbfiles(dstpath, offset=offset, scale=scale)
    except:
        print("create colmap input failed, better clean the data and try again")
        quit()

    for offset in range(startframe, endframe):
        getcolmapsingleimundistort(str(dstpath), offset=offset)

