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
import natsort 

import struct
import pickle
from scipy.spatial.transform import Rotation
from thirdparty.gaussian_splatting.utils.my_utils import rotmat2qvec
from thirdparty.colmap.pre_colmap import * 
from thirdparty.gaussian_splatting.helper3dg import getcolmapsingleimundistort
import argparse

SCALEDICT = {}

# SCALEDICT["01_Welder_S11"] = 0.35
# ["04_Truck", "09_Alexa", "10_Alexa", "11_Alexa", "12_Cave"]
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
    SCALEDICT[scene + "_dist"] = 1.0  # 
    immmersivescaledict[scene + "_dist"] = 1.0



def extractframes(videopath):
    cam = cv2.VideoCapture(videopath)
    ctr = 0
    while ctr < 300:
        _, frame = cam.read()

        savepath = os.path.join(videopath.replace(".mp4", ""), str(ctr) + ".png")
        if not os.path.exists(videopath.replace(".mp4", "")) :
            os.makedirs(videopath.replace(".mp4", ""))
        cv2.imwrite(savepath, frame)
        ctr += 1 
    cam.release()
    return




def extractframesx1(videopath):
    cam = cv2.VideoCapture(videopath)
    ctr = 0
    sucess = True
    while ctr < 300:
        try:
            _, frame = cam.read()

            savepath = os.path.join(videopath.replace(".mp4", ""), str(ctr) + ".png")
            if not os.path.exists(videopath.replace(".mp4", "")) :
                os.makedirs(videopath.replace(".mp4", ""))


            cv2.imwrite(savepath, frame)
            ctr += 1 
        except:
            sucess = False
            cam.release()
            return
    
    cam.release()
    return





def convertmodel2dbfiles(path, offset=0, scale=1.0):
    projectfolder = os.path.join(path, "colmap_" + str(offset))
    #sparsefolder = os.path.join(projectfolder, "sparse/0")
    manualfolder = os.path.join(projectfolder, "manual")

    # if not os.path.exists(sparsefolder):
    #     os.makedirs(sparsefolder)
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
        Knew[0,2] = view['principal_point'][0] 
        Knew[1,2] = view['principal_point'][1] 


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

    postfix  = "_undist" # undistored cameras

    scene = srcscene + postfix
    originalpath = videopath #
    originalvideo = originalpath# 43 1
    path = videopath[:-1] + postfix
    video = originalpath  # 43 1 
    scale = immmersivescaledict[scene]
        


    videoslist = glob.glob(originalvideo + "*.mp4")
    for v in tqdm.tqdm(videoslist):
        extractframesx1(v)


    softlinkdataset(originalpath, path, srcscene, scene)

    try:
        imageundistort(video, offsetlist=[i for i in range(startframe,endframe)],focalscale=scale, fixfocal=None)
    except:
        print("undistort failed")
        quit()
    
    try:
        for offset in tqdm.tqdm(range(startframe, endframe)):
            convertmodel2dbfiles(video, offset=offset, scale=scale)
    except:
        print("create colmap input failed, better clean the data and try again")
        quit()

    for offset in range(0, 50):
        getcolmapsingleimundistort(video, offset=offset)

