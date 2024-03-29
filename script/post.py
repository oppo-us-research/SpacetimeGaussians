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

import natsort 
import os
import glob 
import cv2
import numpy as np
# import sys 
# sys.path.append(".")
from thirdparty.colmap.pre_colmap import * 
from thirdparty.gaussian_splatting.colmap_loader import read_extrinsics_binary, read_intrinsics_binary

def compareSpatialtemporal():
    methoddict = {}
    methoddict["ours"] =  ""
    methoddict["ourslite"] =  ""
    methoddict["dynamic3DGs"] = ""
    methoddict["hpreal"] = "" 
    methoddict["kplane"] = "" 
    methoddict["mixvoxel"] = "" 
    methoddict["nerfplayer"] = "" 

    methoddict["gt"] =    methoddict["ours"].replace("/renders/", "/gt/")

    assert methoddict["ours"] !=  methoddict["gt"]
    

    topleft = (0, 0)   # 
    y, x = topleft[0], topleft[1]

    deltay = 150
    deltaxfames = 250
    
    for k in methoddict.keys():
        total = []
        path = methoddict[k]
        if path != None :
            imagelist = glob.glob(path)
            imagelist = natsort.natsorted(imagelist)
            print(k, len(imagelist))
            imagelist = imagelist[50:]
            print(imagelist[0])
            for imagepath in imagelist[0:deltaxfames]:
                image = cv2.imread(imagepath)
                patch = image[y:y+deltay, x:x+1,:]
                total.append(patch)
            final = np.hstack(total)
            cv2.imwrite("output" + str(k) + ".png", final)

def convertvideos():
    savedir = "/home/output"
    path = "/renders/*.png"

    images = natsort.natsorted(glob.glob(path ))
 
    
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    file_name = os.path.join(savedir, 'flame_steak.mp4')
    video = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height)) # change fps by yourself

    for imageapth in images:
        image = cv2.imread(imageapth)
        video.write(image)

    cv2.destroyAllWindows()
    video.release() 
    # ffmpeg -f image2 -framerate 30 -i img/%05d.png -filter:v "crop=a:b:c:d" -vcodec libx264 -crf 10 xxx.mp4 -pix_fmt yuv420p
# code to look for mached image
def lookforiamge():
    sourceimage = "xxx.jpg"
    image = cv2.imread(sourceimage)
    imagelist = glob.glob("/home/gt/*.png")

    imagelist = natsort.natsorted(imagelist)
    image = cv2.resize(image.astype(np.float32),  (1352, 1014),interpolation=cv2.INTER_CUBIC)
    maxpsnr = 0
    maxpsnrpath = 0
    secondpath = 0
    secondpsnr = 0
    for imagepath in imagelist:
        img2 = cv2.imread(imagepath).astype(np.float32)
        img2 =  cv2.resize(img2.astype(np.float32),  (1352, 1014), interpolation=cv2.INTER_CUBIC)
         
        psnr = cv2.PSNR(image, img2)
        if psnr > maxpsnr:
            secondpsnr = maxpsnr

            maxpsnr = psnr 
            secondpath = maxpsnrpath
            maxpsnrpath = imagepath
            
    
    print(maxpsnr,maxpsnrpath )
    print(secondpsnr,secondpath )


def removenfs():
    # remove not used nfs files
    nfslist = glob.glob("//.nfs*")

    for f in nfslist:
        cmd = " lsof -t " + f +" | xargs kill -9 "
        ret = os.system(cmd)
        print(cmd)

def extractcolmapmodel2db(path, offset=1):
    # 

    projectfolder = os.path.join(path, "colmap_" + str(offset))
    refprojectfolder =  os.path.join(path, "colmap_" + str(0))
    manualfolder = os.path.join(projectfolder, "manual")

    
    cameras_extrinsic_file = os.path.join(refprojectfolder, "sparse/0/", "images.bin") # from distorted?
    cameras_intrinsic_file = os.path.join(refprojectfolder, "sparse/0/", "cameras.bin")

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

    videopath = glob.glob(refprojectfolder + "/images/*.png")
    for i in range(len(videopath)):
        cameraname = os.path.basename(videopath[i])[:-4]#"cam" + str(i).zfill(2)
        cameranameaskey = cameraname + ".png"
        extr, intr =  helperdict[cameranameaskey] # extr.name

        width, height, params = intr.width, intr.height, intr.params
        focolength = intr.params[0]
        fw, fh = intr.params[2], intr.params[3]

        colmapQ = extr.qvec
        T = extr.tvec
        

        imageid = str(i+1)
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

        

        #model, width, height, params = i, W, H, np.array((focolength, W//2, H//2, 0.1))

        camera_id = db.add_camera(1, width, height, params)
        cameraline = str(i+1) + " " + "PINHOLE " + str(width) +  " " + str(height) + " " + str(focolength) + " " + str(focolength) + " " + str(fw) + " " + str(fh) + "\n"
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








WEIGHTDICT={}
DATADICT={}
n3d =  ["flame_salmon_1","flame_steak","cook_spinach", "cut_roasted_beef", "coffee_martini", "sear_steak"]
Techni = ["Birthday", "Fabien", "Painter", "Theater", "Train"]
IMdist = []
CONFIGDICT = {}


for k in n3d:
    WEIGHTDICT[k]= "/"
    DATADICT[k] = "/"

for k in Techni:
    WEIGHTDICT[k]= "/techni/"
    DATADICT[k] = "/technicolor/"

for k in IMdist:
    WEIGHTDICT[k]= "/IMdist/"
    DATADICT[k] = "/IMdist/"
    
def get_value_from_args(args_str, key):
    args_list = args_str.split()
    try:
        key_index = args_list.index(key)
        return args_list[key_index + 1]
    except (ValueError, IndexError):
        return None


def generatescript(gpulist, gpuserver, framerange=[0, 50], step=50, scenelist=["flame_salmon_1"],  option="train",testiter=30000, spetialname="", additional="", add='w', script="trainv2",densifydict=None, cofigroot=None):

    if scenelist[0] not in Techni:
        traincommand = " python " + script + ".py -r 2 --quiet --eval --test_iterations -1" 
        testcommand = " python test.py -r 2" +" --quiet --eval --skip_train"
    else:
        traincommand = " python " + script + ".py --quiet --eval --test_iterations -1" 
        testcommand = " python test.py" +" --quiet --eval --skip_train"
    cmdlist = []


    if option == "gmodel":
        for scene in scenelist:
            for i in framerange:
                colmapfolder = DATADICT[scene]+ scene +"/colmap_" + str(i)
                modelsavefolder =WEIGHTDICT[scene] + spetialname + "/" + scene +"/colmap_" + str(i)
                tmpcommdad = traincommand + " -s " + colmapfolder + " -m " + modelsavefolder  + " --config "+  cofigroot + scene + ".json" 
                curtestcommand = testcommand + " -s " + colmapfolder + " -m " + modelsavefolder  + " --config "+ cofigroot + scene + ".json" # how to overwrite config with additional input?
      
                try:
                    if scene in densifydict:
                        tmpcommdad += " --densify " + str(densifydict[scene])
                except:
                    pass
      
                curtestcommand += " --test_iteration " + str(testiter)

                cmdlist.append((tmpcommdad, curtestcommand))

    sequences = np.array_split(cmdlist, len(gpulist))

    if scene in n3d:
        additonaltest = " --valloader colmapvalid"
    elif scene in Techni:
        additonaltest = " --valloader technicolor"
    elif scene in IMdist:
        additonaltest = " --valoader immersivevalidss"
    for idx in range(len(gpulist)):
        gpuid = gpulist[idx]
        scriptname = gpuserver + '_gpu' + str(gpuid).zfill(1) + option +".sh" # to acrross different server 
        thisfilecommands =  sequences[idx]
        print("writing to ", scriptname)
        with open(scriptname, add) as filescript:
            if len(thisfilecommands[0]) == 3:
                for op, testop, metricop in thisfilecommands:
                    train = "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=" + str(gpuid) +  " "+ op + " " + additional # PYTHONDONTWRITEBYTECODE=1 important !
                    test = "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=" + str(gpuid) +  " "+ testop  + additonaltest
                    if testiter == 30000:
                        filescript.write("%s ;\n" % train)
                        filescript.write("%s ;\n" % test) # skip test f
            else :
              for op in thisfilecommands:
                    train = "CUDA_VISIBLE_DEVICES=" + str(gpuid) +  " "+ op 
                    filescript.write("%s ;\n" % train)

            filescript.close()



if __name__ == "__main__" :
    #removenfs()
    #compareSpatialtemporal()
    # convertvideos()
    pass # 
