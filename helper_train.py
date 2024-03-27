#
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

# =============================================

# This license is additionally subject to the following restrictions:

# Licensor grants non-exclusive rights to use the Software for research purposes
# to research users (both academic and industrial), free of charge, without right
# to sublicense. The Software may be used "non-commercially", i.e., for research
# and/or evaluation purposes only.

# Subject to the terms and conditions of this License, you are granted a
# non-exclusive, royalty-free, license to reproduce, prepare derivative works of,
# publicly display, publicly perform and distribute its Work and any resulting
# derivative works in any form.
#

import torch
import numpy as np
import torch
from simple_knn._C import distCUDA2
import os 
import json 
import cv2
from script.pre_immersive_distorted import SCALEDICT 


def getrenderpip(option="train_ours_full"):
    print("render option", option)
    if option == "train_ours_full":
        from thirdparty.gaussian_splatting.renderer import train_ours_full 
        from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch9 import GaussianRasterizer  
        return train_ours_full, GaussianRasterizationSettings, GaussianRasterizer


    elif option == "train_ours_lite":
        from thirdparty.gaussian_splatting.renderer import train_ours_lite
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer  

        return train_ours_lite, GaussianRasterizationSettings, GaussianRasterizer
    
    elif option == "test_ours_full":
        from thirdparty.gaussian_splatting.renderer import test_ours_full
        from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings 
        from diff_gaussian_rasterization_ch9 import GaussianRasterizer  
        return test_ours_full, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_ours_lite": # forward only 
        from thirdparty.gaussian_splatting.renderer import test_ours_lite
        from forward_lite import GaussianRasterizationSettings 
        from forward_lite import GaussianRasterizer 
        return test_ours_lite, GaussianRasterizationSettings, GaussianRasterizer
    
    elif option == "export_to_unity":
        from thirdparty.gaussian_splatting.renderer import export_to_unity
        from forward_lite import GaussianRasterizationSettings 
        from forward_lite import GaussianRasterizer 
        return export_to_unity, GaussianRasterizationSettings, GaussianRasterizer
    else:
        raise NotImplementedError("Rennder {} not implemented".format(option))
    
def getmodel(model="oursfull"):
    if model == "ours_full":
        from  thirdparty.gaussian_splatting.scene.oursfull import GaussianModel
    elif model == "ours_lite":
        from  thirdparty.gaussian_splatting.scene.ourslite import GaussianModel
    else:
    
        raise NotImplementedError("model {} not implemented".format(model))
    return GaussianModel

def getloss(opt, Ll1, ssim, image, gt_image, gaussians, radii):
    if opt.reg == 1: # add optical flow loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians._motion) / gaussians._motion.shape[0]
    elif opt.reg == 0 :
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))   
    elif opt.reg == 9 : #regulizor on the rotation
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians._omega[radii>0]**2)
    elif opt.reg == 10 : #regulizor on the rotation
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians._motion[radii>0]**2)
    elif opt.reg == 4:
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(gaussians.get_scaling) / gaussians._motion.shape[0]
    elif opt.reg == 5:
        loss = Ll1  
    elif opt.reg == 6 :
        ratio = torch.mean(gt_image) - 0.5 + opt.lambda_dssim
        ratio = torch.clamp(ratio, 0.0, 1.0)
        loss = (1.0 - ratio) * Ll1 + ratio * (1.0 - ssim(image, gt_image))
    elif opt.reg == 7 :
        Ll1 = Ll1 / (torch.mean(gt_image) * 2.0) # normalize L1 loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    elif opt.reg == 8 :
        N = gaussians._xyz.shape[0]
        mean = torch.mean(gaussians._xyz, dim=0, keepdim=True)
        varaince = (mean - gaussians._xyz)**2 #/ N
        loss = (1.0 - opt.lambda_dssim) * Ll1  + 0.0002*opt.lambda_dssim * torch.sum(varaince) / N
    return loss 


def freezweights(model, screenlist):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        newgrad = torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return  

def freezweightsbymask(model, screenlist, mask):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        newgrad =  mask.unsqueeze(1)*grad_tensor #torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return  


def freezweightsbymasknounsqueeze(model, screenlist, mask):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        newgrad =  mask*grad_tensor #torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return  


def removeminmax(gaussians, maxbounds, minbounds):
    maxx, maxy, maxz = maxbounds
    minx, miny, minz = minbounds
    xyz = gaussians._xyz
    mask0 = xyz[:,0] > maxx.item()
    mask1 = xyz[:,1] > maxy.item()
    mask2 = xyz[:,2] > maxz.item()

    mask3 = xyz[:,0] < minx.item()
    mask4 = xyz[:,1] < miny.item()
    mask5 = xyz[:,2] < minz.item()
    mask =  logicalorlist([mask0, mask1, mask2, mask3, mask4, mask5])
    gaussians.prune_points(mask) 
    torch.cuda.empty_cache()


def controlgaussians(opt, gaussians, densify, iteration, scene,  visibility_filter, radii, viewspace_point_tensor, flag, traincamerawithdistance=None, maxbounds=None, minbounds=None): 
    if densify == 1: # n3d 
        if iteration < opt.densify_until_iter :
            if iteration ==  8001 : # 8001
                omegamask = gaussians.zero_omegabymotion() # 1 we keep omega, 0 we freeze omega
                gaussians.omegamask  = omegamask
                scene.recordpoints(iteration, "seperate omega"+str(torch.sum(omegamask).item()))
            elif iteration > 8001: # 8001
                freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
                rotationmask = torch.logical_not(gaussians.omegamask)
                freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask)
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if flag < opt.desicnt:
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    flag+=1
                    scene.recordpoints(iteration, "after densify")
                else:
                    if iteration < 7000 : # defalt 7000. 
                        prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                        gaussians.prune_points(prune_mask)
                        torch.cuda.empty_cache()
                        scene.recordpoints(iteration, "addionally prune_mask")
            if iteration % 3000 == 0 :
                gaussians.reset_opacity()
        else:
            freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
            rotationmask = torch.logical_not(gaussians.omegamask)
            freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask) #uncomment freezeweight... for fast traning speed.
            if iteration % 1000 == 500 :
                zmask = gaussians._xyz[:,2] < 4.5  # 
                gaussians.prune_points(zmask) 
                torch.cuda.empty_cache()
            if iteration == 10000: 
                removeminmax(gaussians, maxbounds, minbounds)
        return flag
    
    elif densify == 2: # n3d 
        if iteration < opt.densify_until_iter :
            if iteration ==  8001 : # 8001
                omegamask = gaussians.zero_omegabymotion() #
                gaussians.omegamask  = omegamask
                scene.recordpoints(iteration, "seperate omega"+str(torch.sum(omegamask).item()))
            elif iteration > 8001: # 8001
                freezweightsbymasknounsqueeze(gaussians, ["_omega"], gaussians.omegamask)
                rotationmask = torch.logical_not(gaussians.omegamask)
                freezweightsbymasknounsqueeze(gaussians, ["_rotation"], rotationmask)
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if flag < opt.desicnt:
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    flag+=1
                    scene.recordpoints(iteration, "after densify")
                else:
                    prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                    gaussians.prune_points(prune_mask)
                    torch.cuda.empty_cache()
                    scene.recordpoints(iteration, "addionally prune_mask")
            if iteration % 3000 == 0 :
                gaussians.reset_opacity()
        else:
            if iteration % 1000 == 500 :
                zmask = gaussians._xyz[:,2] < 4.5  # for stability  
                gaussians.prune_points(zmask) 
                torch.cuda.empty_cache()
        return flag
    

    elif densify == 3: # techni
        if iteration < opt.densify_until_iter :
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if flag < opt.desicnt:
                    scene.recordpoints(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_pruneclone(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
                    flag+=1
                    scene.recordpoints(iteration, "after densify")
                else:
                    if iteration < 7000 : # defalt 7000. 
                        prune_mask =  (gaussians.get_opacity < opt.opthr).squeeze()
                        gaussians.prune_points(prune_mask)
                        torch.cuda.empty_cache()
                        scene.recordpoints(iteration, "addionally prune_mask")
            if iteration % opt.opacity_reset_interval == 0 :
                gaussians.reset_opacity()
        else:
            if iteration == 10000: 
                removeminmax(gaussians, maxbounds, minbounds)
        return flag
    


def logicalorlist(listoftensor):
    mask = None 
    for idx, ele in enumerate(listoftensor):
        if idx == 0 :
            mask = ele 
        else:
            mask = torch.logical_or(mask, ele)
    return mask 



def recordpointshelper(model_path, numpoints, iteration, string):
    txtpath = os.path.join(model_path, "exp_log.txt")
    
    with open(txtpath, 'a') as file:
        file.write("iteration at "+ str(iteration) + "\n")
        file.write(string + " pointsnumber " + str(numpoints) + "\n")




def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0




def reloadhelper(gaussians, opt, maxx, maxy, maxz,  minx, miny, minz):
    givenpath = opt.prevpath
    if opt.loadall == 0:
        gaussians.load_plyandminmax(givenpath, maxx, maxy, maxz,  minx, miny, minz)
    elif opt.loadall == 1 :
        gaussians.load_plyandminmaxall(givenpath, maxx, maxy, maxz,  minx, miny, minz)
    elif opt.loadall == 2 :        
        gaussians.load_ply(givenpath)
    elif opt.loadall == 3:
        gaussians.load_plyandminmaxY(givenpath, maxx, maxy, maxz,  minx, miny, minz)

    gaussians.max_radii2D =  torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    return 

def getfisheyemapper(folder, cameraname):
    parentfolder = os.path.dirname(folder)
    distoritonflowpath = os.path.join(parentfolder, cameraname + ".npy")
    distoritonflow = np.load(distoritonflowpath)
    distoritonflow = torch.from_numpy(distoritonflow).unsqueeze(0).float().cuda()
    return distoritonflow











def undistortimage(imagename, datasetpath,data):
    


    video = os.path.dirname(datasetpath) # upper folder 
    with open(os.path.join(video + "/models.json"), "r") as f:
                meta = json.load(f)

    for idx , camera in enumerate(meta):
        folder = camera['name'] # camera_0001
        view = camera
        intrinsics = np.array([[view['focal_length'], 0.0, view['principal_point'][0]],
                            [0.0, view['focal_length'], view['principal_point'][1]],
                            [0.0, 0.0, 1.0]])
        dis_cef = np.zeros((4))

        dis_cef[:2] = np.array(view['radial_distortion'])[:2]
        if folder != imagename:
             continue
        print("done one camera")
        map1, map2 = None, None
        sequencename = os.path.basename(video)
        focalscale = SCALEDICT[sequencename]
 
        h, w = data.shape[:2]


        image_size = (w, h)
        knew = np.zeros((3, 3), dtype=np.float32)

def trbfunction(x): 
    return torch.exp(-1*x.pow(2))
