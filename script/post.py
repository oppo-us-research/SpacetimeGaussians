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
    nfslist = glob.glob("/.nfs*")

    for f in nfslist:
        cmd = " lsof -t " + f +" | xargs kill -9 "
        ret = os.system(cmd)
        print(cmd)

if __name__ == "__main__" :
    #compareSpatialtemporal()
    # convertvideos()
    pass # 
