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
import cv2 
import glob 
import tqdm 
import numpy as np 
import shutil
import pickle
import argparse


import natsort 

import struct
import pickle
from scipy.spatial.transform import Rotation
from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec, qvec2rotmat
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov, fov2focal
from thirdparty.colmap.pre_colmap import *
from thirdparty.gaussian_splatting.helper3dg import getcolmapsingleimdistort 
from script.pre_n3d import extractframes
SCALEDICT = {}

# SCALEDICT["01_Welder_S11"] = 0.35
# ["04_Truck", "09_Alexa", "10_Alexa", "11_Alexa", "12_Cave"]
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


        



def convertmodel2dbfiles(path, offset=0, scale=1.0, removeverythingexceptinput=False):


    projectfolder = os.path.join(path, "colmap_" + str(offset))
    manualfolder = os.path.join(projectfolder, "manual")

    
    if os.path.exists(projectfolder) and removeverythingexceptinput:
        print("already exists colmap folder, better remove it and create a new one")
        inputfolder = os.path.join(projectfolder, "input")
        # remove everything except input folder
        for file in os.listdir(projectfolder):
            if file == "input":
                continue
            file_path = os.path.join(projectfolder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    if not os.path.exists(manualfolder):
        os.makedirs(manualfolder)

    savetxt = os.path.join(manualfolder, "images.txt")
    savecamera = os.path.join(manualfolder, "cameras.txt")
    savepoints = os.path.join(manualfolder, "points3D.txt")
    imagetxtlist = []
    cameratxtlist = []
    if os.path.exists(os.path.join(projectfolder, "input.db")):
        os.remove(os.path.join(projectfolder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(projectfolder, "input.db"))

    db.create_tables()



    import json 
    with open(os.path.join(video + "models.json"), "r") as f:
        meta = json.load(f)

    for idx , camera in enumerate(meta):
        cameraname = camera['name'] # camera_0001
        view = camera

 
        focolength = camera['focal_length'] 
        width, height = camera['width'], camera['height']
        principlepoint =[0,0]
        principlepoint[0] = view['principal_point'][0]
        principlepoint[1] = view['principal_point'][1]



 

        distort1 = view['radial_distortion'][0]
        distort2 = view['radial_distortion'][1]
        distort3 = 0
        distort4 = 0 #view['radial_distortion'][3]


        R = Rotation.from_rotvec(view['orientation']).as_matrix()
        t = np.array(view['position'])[:, np.newaxis]
        w2c = np.concatenate((R, -np.dot(R, t)), axis=1)
        
        colmapR = w2c[:3, :3]
        T = w2c[:3, 3]



        K = np.array([[focolength, 0, principlepoint[0]], [0, focolength, principlepoint[1]], [0, 0, 1]])
        Knew = K.copy()
        
        Knew[0,0] = K[0,0] * float(scale)
        Knew[1,1] = K[1,1] * float(scale)
        Knew[0,2] = view['principal_point'][0]#width * 0.5 #/ 2
        Knew[1,2] = view['principal_point'][1]#height * 0.5 #/ 2

        # transformation = np.array([[2,   0.0, 0.5],
        #                            [0.0, 2,   0.5],
        #                            [0.0, 0.0, 1.0]])
        # Knew = np.dot(transformation, Knew)

        newfocalx = Knew[0,0]
        newfocaly = Knew[1,1]
        newcx = Knew[0,2]
        newcy = Knew[1,2]



        colmapQ = rotmat2qvec(colmapR)

        imageid = str(idx+1)
        cameraid = imageid
        pngname = cameraname + ".png"
        
        line =  imageid + " "

        for j in range(4):
            line += str(colmapQ[j]) + " "
        for j in range(3):
            line += str(T[j]) + " "
        line = line  + cameraid + " " + pngname + "\n"
        empltyline = "\n"
        imagetxtlist.append(line)
        imagetxtlist.append(empltyline)

        newwidth = width
        newheight = height
        params = np.array((newfocalx , newfocaly, newcx, newcy,))

        camera_id = db.add_camera(1, newwidth, newheight, params)     # RADIAL_FISHEYE                                                                                 # width and height
        #
        #cameraline = str(i+1) + " " + "PINHOLE " + str(width) +  " " + str(height) + " " + str(focolength) + " " + str(focolength) + " " + str(W//2) + " " + str(H//2) + "\n"

        cameraline = str(idx+1) + " " + "PINHOLE " + str(newwidth) +  " " + str(newheight) + " " + str(newfocalx) + " " + str(newfocaly)  + " " + str(newcx) + " " + str(newcy)  + "\n"
        cameratxtlist.append(cameraline)
        image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((T[0], T[1], T[2])), image_id=idx+1)
        db.commit()
        print("commited one")
    db.close()


    with open(savetxt, "w") as f:
        for line in imagetxtlist :
            f.write(line)
    with open(savecamera, "w") as f:
        for line in cameratxtlist :
            f.write(line)
    with open(savepoints, "w") as f:
        pass 




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
    with open(os.path.join(video + "models.json"), "r") as f:
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
                #
                distortingflow = getdistortedflow(image, intrinsics, dis_cef, "linear", crop_output=False, scale=1.0, knew=knew)
                print("saved distortion mappers")
                np.save(os.path.join(video, folder  + ".npy"), distortingflow)






def softlinkdataset(originalpath, path, srcscene, scene):
    videofolderlist = glob.glob(originalpath + "camera_*/")

    if not os.path.exists(path):
        os.makedirs(path)

    for videofolder in videofolderlist:
        newlink = os.path.join(path, videofolder.split("/")[-2])
        if os.path.exists(newlink):
            print("already exists do not make softlink again")
            quit()
        assert not os.path.exists(newlink)
        cmd = " ln -s " + videofolder + " " + newlink
        os.system(cmd)
        print(cmd)

    originalmodel = originalpath + "models.json"
    newmodel = path + "models.json"
    shutil.copy(originalmodel, newmodel)
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
    path = videopath[:-1] + postfix
    video = originalpath  # 43 1 
    scale = immmersivescaledict[scene]


    videoslist = glob.glob(originalvideo + "*.mp4")
    for v in tqdm.tqdm(videoslist):
        extractframes(v)

    try:
        softlinkdataset(originalpath, path, srcscene, scene)
    except:
        print("softlink failed")
        quit()
    try:
        imageundistort(video, offsetlist=[i for i in range(startframe,endframe)],focalscale=scale, fixfocal=None)
    except:
        print("undistort failed")
        quit()

    try:
        for offset in tqdm.tqdm(range(startframe, endframe)):
            convertmodel2dbfiles(video, offset=offset, scale=scale, removeverythingexceptinput=False)
    except:
        convertmodel2dbfiles(video, offset=offset, scale=scale, removeverythingexceptinput=True)
        print("create colmap input failed, better clean the data and try again")
        quit()
    for offset in range(startframe, endframe):
        getcolmapsingleimdistort(video, offset=offset)
