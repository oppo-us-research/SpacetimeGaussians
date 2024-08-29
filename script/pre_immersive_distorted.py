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
import argparse
import cv2
import numpy as np
import os 
import json 

import natsort 
import sys
import struct
import pickle
from scipy.spatial.transform import Rotation
import sys 

from script.utils_pre import write_colmap

sys.path.append(".")
from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec, qvec2rotmat
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov, fov2focal
from thirdparty.colmap.pre_colmap import *
from thirdparty.gaussian_splatting.helper3dg import getcolmapsingleimdistort 
from script.pre_n3d import extractframes
SCALEDICT = {}


Immersiveseven = ["01_Welder",  "02_Flames", "04_Truck", "09_Alexa", "10_Alexa", "11_Alexa", "12_Cave"]
immmersivescaledict = {}
immmersivescaledict["01_Welder"] = 0.36
immmersivescaledict["02_Flames"] = 0.35
immmersivescaledict["04_Truck"] = 0.36
immmersivescaledict["09_Alexa"] = 0.36
immmersivescaledict["10_Alexa"] = 0.36
immmersivescaledict["11_Alexa"] = 0.36
immmersivescaledict["12_Cave"] = 0.36

for scene in Immersiveseven:
    immmersivescaledict[scene + "_dist"] =immmersivescaledict[scene] 
    SCALEDICT[scene + "_dist"] = immmersivescaledict[scene]  #immmersivescaledict[scene]  # to be checked with large scale


def convertmodel2dbfiles(path, offset=0, scale=1.0):
    with (path / "models.json").open("r") as f:
        meta = json.load(f)

    cameras =[]

    for idx , camera in enumerate(meta):
        focolength = camera['focal_length'] 
        R = Rotation.from_rotvec(camera['orientation']).as_matrix()
        t = np.array(camera['position'])[:, np.newaxis]
        w2c = np.concatenate((R, -np.dot(R, t)), axis=1)
        
        colmapR = w2c[:3, :3]
        T = w2c[:3, 3]
        colmapQ = rotmat2qvec(colmapR)

        cameras.append({
            'id': str(idx+1),
            'filename': camera['name']+'.png',
            'w': camera['width'],
            'h': camera['height'],
            'fx': focolength * float(scale),
            'fy': focolength * float(scale),
            'cx': camera['principal_point'][0],
            'cy': camera['principal_point'][1],
            'q': colmapQ,
            't': T,
        })

    write_colmap(path, cameras, offset)



#https://github.com/Synthesis-AI-Dev/fisheye-distortion
def getdistortedflow(img: np.ndarray, cam_intr: np.ndarray, dist_coeff: np.ndarray,
                  mode: str, crop_output: bool = True,
                  crop_type: str = "corner", scale: float =2, cxoffset=None, cyoffset=None, knew=None):
 
    
    assert cam_intr.shape == (3, 3)
    assert dist_coeff.shape == (4,)

    imshape = img.shape
    if len(imshape) == 3:
        h, w, chan = imshape
    elif len(imshape) == 2:
        h, w = imshape
        chan = 1
    else:
        raise RuntimeError(f'Image has unsupported shape: {imshape}. Valid shapes: (H, W), (H, W, N)')
    
    imdtype = img.dtype
    dstW = int(w )
    dstH = int(h )

    # Get array of pixel co-ords
    xs = np.arange(dstW)
    ys = np.arange(dstH)

    xs = xs #- 0.5 # + cxoffset / 2 
    ys = ys #- 0.5 # + cyoffset / 2 
    
    xv, yv = np.meshgrid(xs, ys)
    img_pts = np.stack((xv, yv), axis=2)  # shape (H, W, 2)
    img_pts = img_pts.reshape((-1, 1, 2)).astype(np.float32)  # shape: (N, 1, 2), in undistorted image coordiante

    
    undistorted_px = cv2.fisheye.undistortPoints(img_pts, cam_intr, dist_coeff, None, knew)  # shape: (N, 1, 2)
    

    undistorted_px = undistorted_px.reshape((dstH, dstW, 2))  # Shape: (H, W, 2)
    undistorted_px = np.flip(undistorted_px, axis=2)  # flip x, y coordinates of the points as cv2 is height first
    
    
    undistorted_px[:, :, 0] = undistorted_px[:, :, 0]  #+  0.5*cyoffset #- 0.25*cyoffset #orginalx (0, 1)
    undistorted_px[:, :, 1] = undistorted_px[:, :, 1]  #+  0.5*cyoffset #- 0.25*cxoffset #orginaly (0, 1)

    
    undistorted_px[:, :, 0] = undistorted_px[:, :, 0] / (h-1)#(h-1) #orginalx (0, 1)
    undistorted_px[:, :, 1] = undistorted_px[:, :, 1] / (w-1)#(w-1) #orginaly (0, 1)


    undistorted_px = 2 * (undistorted_px - 0.5)           #to -1 to 1 for gridsample
    

    undistorted_px[:, :, 0] = undistorted_px[:, :, 0]  #orginalx (0, 1)
    undistorted_px[:, :, 1] = undistorted_px[:, :, 1]  #orginaly (0, 1)
    
    
    undistorted_px = undistorted_px[:,:,::-1] # yx to xy for grid sample
    return undistorted_px
    


def imageundistort(video, offsetlist=[0],focalscale=1.0, fixfocal=None):
    import cv2
    import numpy as np
    import os 
    import json 
    with open(os.path.join(video, "models.json"), "r") as f:
                meta = json.load(f)

    for idx , camera in enumerate(meta):
        folder = camera['name'] # camera_0001
        view = camera
        intrinsics = np.array([[view['focal_length'], 0.0, view['principal_point'][0]],
                            [0.0, view['focal_length'], view['principal_point'][1]],
                            [0.0, 0.0, 1.0]])
        dis_cef = np.zeros((4))

        dis_cef[:2] = np.array(view['radial_distortion'])[:2]
        print("done one camera")
        map1, map2 = None, None
        for offset in offsetlist:
            videofolder = os.path.join(video, folder)
            imagepath = os.path.join(videofolder, str(offset) + ".png")
            imagesavepath = os.path.join(video, "colmap_" + str(offset), "input", folder + ".png")
            if os.path.exists(imagesavepath):
                pass
            else:
                inputimagefolder = os.path.join(video, "colmap_" + str(offset), "input")
                if not os.path.exists(inputimagefolder):
                    os.makedirs(inputimagefolder)
                assert os.path.exists(imagepath)
                image = cv2.imread(imagepath).astype(np.float32) #/ 255.0
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

            if offset == 0:
                distortionmapperpath = os.path.join(video, folder  + ".npy")
                 
                if os.path.exists(distortionmapperpath):
                    print("already exists mapper")
                    pass 
                else:
                    distortingflow = getdistortedflow(image, intrinsics, dis_cef, "linear", crop_output=False, scale=1.0, knew=knew)
                    print("saved distortion mappers")
                    np.save(os.path.join(video, folder  + ".npy"), distortingflow)






def softlinkdataset(original_str, target_str):
    originalpath = Path(original_str)
    path = Path(target_str)

    videofolderlist = [f for f in sorted(originalpath.glob("camera_*")) if f.is_dir()]
    path.mkdir(exist_ok=True)
    for videofolder in videofolderlist:
        newlink = path / videofolder.name
        if not newlink.exists():
            newlink.symlink_to(videofolder.resolve()) #make sure to use absolute path with symlink
            print(f"symlink: {newlink} -> {videofolder.resolve()}")
        else:
            print("already exists, do not make softlink again")
    shutil.copy(originalpath / "models.json", path / "models.json")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=50, type=int)


    args = parser.parse_args()
    videopath = args.videopath

    startframe = args.startframe
    endframe = args.endframe
    if startframe >= endframe:
        print("start frame must smaller than end frame")
        quit()
    if startframe < 0 or endframe > 300:
        print("frame must in range 0-300")
        quit()
    if not os.path.exists(videopath):
        print("path not exist")
        quit()
    
    if not videopath.endswith("/"):
        videopath = videopath + "/"

    srcscene = videopath.split("/")[-2]
    if srcscene not in Immersiveseven:
        print("scene not in Immersiveseven", Immersiveseven)
        print("Please check if the scene name is correct")
        quit()
    

    if "04_Trucks" in videopath:
        print('04_Trucks')
        if endframe > 150:
            endframe = 150 

    postfix  = "_dist" # distored model

    scene = srcscene + postfix
    originalpath = videopath #" 
    originalvideo = originalpath# 43 1
    dstpath = videopath[:-1] + postfix # the path to save the dataset.

    scale = immmersivescaledict[scene]


    videoslist = glob.glob(originalvideo + "*.mp4")
    for v in tqdm.tqdm(videoslist):
        extractframes(Path(v))

    softlinkdataset(originalpath, dstpath)
  
    
    imageundistort(dstpath, offsetlist=[i for i in range(startframe,endframe)],focalscale=scale, fixfocal=None)


    try:
        for offset in tqdm.tqdm(range(startframe, endframe), desc="convertmodel2dbfiles"):
            convertmodel2dbfiles(Path(dstpath), offset=offset, scale=scale)
    except:
        print("create colmap input failed, better clean the data and try again")
        quit()
    for offset in range(startframe, endframe):
        getcolmapsingleimdistort(dstpath, offset=offset)
