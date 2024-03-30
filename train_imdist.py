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
# ========================================================================================================
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the thirdparty/gaussian_splatting/LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
import random 
import sys
import uuid
import json 


sys.path.append("./thirdparty/gaussian_splatting")
### do no
from thirdparty.gaussian_splatting.utils.loss_utils import l1_loss, ssim, l2_loss, rel_loss, ssimmap
from helper_train import getrenderpip, getmodel, getloss, removeminmax, reloadhelper, trbfunction,undistortimage
from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.utils.general_utils import safe_state
from tqdm import tqdm
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams



from tqdm import tqdm
import time 
import torchvision
import numpy as np 
import torch.nn.functional as F
import pickle 
import cv2
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
EPS = 1e-6

def freezweightsbymasknounsqueeze(model, screenlist, mask):
    for k in screenlist:
        grad_tensor = getattr(getattr(model, k), 'grad')
        newgrad =  mask*grad_tensor #torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), 'grad', newgrad)
    return  


def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def train(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, densify=0, duration=50, basicfunction="gaussian", rgbfunction="rgbv1", rdpip="v2"):
    first_iter = 0
    render, GRsetting, GRzer = getrenderpip(rdpip)

    tb_writer = prepare_output_and_logger(dataset)
    print("use model {}".format(dataset.model))
    GaussianModel = getmodel(dataset.model) 
    
    gaussians = GaussianModel(dataset.sh_degree, rgbfunction)
    gaussians.trbfslinit = -1*opt.trbfslinit  # control the scale of trbf  
    gaussians.preprocesspoints = opt.preprocesspoints 

    rbfbasefunction = trbfunction
    scene = Scene(dataset, gaussians, duration=duration, loader=dataset.loader)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    numchannel = 9 

    bg_color = [1, 1, 1] if dataset.white_background else [0 for i in range(numchannel)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    #if freeze != 1:
    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    flag = 0
    closethreshold = None
    depthdict = {}

    if opt.batch > 1:
        traincameralist = scene.getTrainCameras().copy()
        traincamdict = {}
        for i in range(duration): # 0 to 4, -> (0.0, to 0.8)
            traincamdict[i] = [cam for cam in traincameralist if cam.timestamp == i/duration]


    scalethreshold = gaussians.percent_dense * scene.cameras_extent  
    
    
    if gaussians.ts is None :
        H,W = traincameralist[0].image_height, traincameralist[0].image_width
        gaussians.ts = torch.ones(1,1,H,W).cuda()

    scene.recordpoints(0, "start training")
    startime = time.time()
    gaussians.raystart = opt.raystart

    currentxyz = gaussians._xyz 

    maxx, maxy, maxz = torch.amax(currentxyz[:,0]), torch.amax(currentxyz[:,1]), torch.amax(currentxyz[:,2])# 
    minx, miny, minz = torch.amin(currentxyz[:,0]), torch.amin(currentxyz[:,1]), torch.amin(currentxyz[:,2])
    
    if os.path.exists(opt.prevpath): # reload trained model to boost results. 
        print("load from " + opt.prevpath)
        reloadhelper(gaussians, opt, maxx, maxy, maxz,  minx, miny, minz)


    maxbounds = [maxx, maxy, maxz]
    minbounds = [minx, miny, minz]

                                                            
    flagems = 0 # chagne to 1 to start ems
    emscnt = 0
    maxloss = None
    maxlosscamera = None
    lossdiect = {}
    ssimdict = {}
    depthdict = {}
    validdepthdict = {}
    emsstartfromiterations = opt.emsstart 
    assert opt.losstart < opt.emsstart
    
    with torch.no_grad():
        timeindex = 0 # 0 to 49
        viewpointset = traincamdict[timeindex]

        for viewpoint_cam in viewpointset:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background,  override_color=None,  basicfunction=rbfbasefunction, GRsetting=GRsetting, GRzer=GRzer)
            
            _, depthH, depthW = render_pkg["depth"].shape
            borderH = int(depthH/2)
            borderW = int(depthW/2)

            midh =  int(viewpoint_cam.image_height/2)
            midw =  int(viewpoint_cam.image_width/2)
            
            depth = render_pkg["depth"]
            slectemask = depth != 15.0 

            validdepthdict[viewpoint_cam.image_name] = depth[slectemask].var().item()
            depthdict[viewpoint_cam.image_name] = torch.amax(depth[slectemask]).item() 
            ssimdict[viewpoint_cam.image_name] =  ssim(render_pkg["render"].detach(), viewpoint_cam.original_image.float().detach()).item()



    orderedlossdiect = sorted(ssimdict.items(), key=lambda item: item[1], reverse=False)
    orderedestph = sorted(validdepthdict.items(), key=lambda item: item[1], reverse=True)

    totalength = len(orderedestph)
    mid = int(totalength/2)
    middepthlist = [p[0] for p in orderedestph[:mid]]#  

    for k in middepthlist:
        scene.recordpoints(0, "selective: " + k )

    midlosslist = [p[0] for p in orderedlossdiect[:mid]]

    datasetroot = os.path.dirname(dataset.source_path)
    pickedviewspath = os.path.join(datasetroot, "pickview.pkl")
    selectviews = midlosslist[1:4] 
    if not os.path.exists(pickedviewspath):
        print("please copy pick view")
        quit()
        # with open(pickedviewspath, 'wb') as handle:
        #     pickle.dump(selectviews, handle, protocol=pickle.HIGHEST_PROTOCOL) # uncomment to dump the selectview to the dataset please select the duration = 1 

    else:
        selectviews = load_pkl(pickedviewspath)

    for k in selectviews:
        scene.recordpoints(0, "load: " + k )


    selectedlength = 3
    lasterems = 0 
    lastrest = 0
    
    
    for iteration in range(first_iter, opt.iterations + 1): 

             
        if iteration ==  opt.emsstart:
            flagems = 2 # start ems



        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera, or fewer than batch to pop
        if opt.batch == 1 and not viewpoint_stack: 
            viewpoint_stack = scene.getTrainCameras().copy()
        
        
        if (iteration - 1) == debug_from:
            pipe.debug = True
        if gaussians.rgbdecoder is not None:
            gaussians.rgbdecoder.train()

        if opt.batch > 1:
            gaussians.zero_gradient_cache()


            timeindex = randint(0, duration-1) # 0 to 49
            viewpointset = traincamdict[timeindex]
            camindex = random.sample(viewpointset, opt.batch)


            for i in range(opt.batch):
                viewpoint_cam = camindex[i]
                render_pkg = render(viewpoint_cam, gaussians, pipe, background,  override_color=None,  basicfunction=rbfbasefunction, GRsetting=GRsetting, GRzer=GRzer)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                gt_image = viewpoint_cam.original_image.float().cuda() 


                
                if opt.reg == 2:
                    Ll1 = l2_loss(image, gt_image)
                    loss = Ll1
                elif opt.reg == 3:
                    Ll1 = rel_loss(image, gt_image)
                    loss = Ll1
                else:
                    Ll1 = l1_loss(image, gt_image)
                    loss = getloss(opt, Ll1, ssim, image, gt_image, gaussians, radii)
                
                loss.backward()
                gaussians.cache_gradient()
                gaussians.optimizer.zero_grad(set_to_none = True)# 



            
            iter_end.record()
            gaussians.set_batch_gradient(opt.batch)


        else:
            raise NotImplementedError("Batch size 1 is not supported anymore")

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save                                                                                                         #viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                durationtime = time.time() - startime
                txtpath = scene.model_path + "/trainingtime.txt"
                with open(txtpath, "w") as f:
                    f.write(str(iteration) + " cost time: "+ str(durationtime))

            # ensure that parameters are same as in the model
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            
            if densify == 4: # 
                if iteration < opt.densify_until_iter :
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
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
                            gaussians.densify_prunecloneim(opt.densify_grad_threshold, op.opthr, scene.cameras_extent, size_threshold)
                            flag+=1
                            scene.recordpoints(iteration, "after densify")
                        else:
                            if iteration < 5000:
                                prune_mask =  (gaussians.get_opacity < op.opthr).squeeze()
                                if opt.prunebysize :
                                    big_points_vs = gaussians.max_radii2D > 20
                                    big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                                    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                                if (iteration > (500 + lastrest)) and lastrest > 1000: #
                                    gaussians.prune_points(prune_mask)
                                    torch.cuda.empty_cache()
                                    scene.recordpoints(iteration, "addionally prune_mask")
                    if iteration % opt.opacity_reset_interval == 0 and iteration < 4000:
                        gaussians.reset_opacity()
                        lastrest = iteration

            if densify == 6: # 
                if iteration < opt.densify_until_iter :
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        if flag < opt.desicnt:
                            scene.recordpoints(iteration, "before densify")
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.densify_prunecloneimgeneral(opt.densify_grad_threshold, op.opthr, scene.cameras_extent, size_threshold)
                            flag+=1
                            scene.recordpoints(iteration, "after densify")
                        else:
                            if iteration < 9000:
                                prune_mask =  (gaussians.get_opacity < op.opthr).squeeze()
                                if opt.prunebysize :
                                    big_points_vs = gaussians.max_radii2D > 20
                                    big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                                    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                                gaussians.prune_points(prune_mask)
                                torch.cuda.empty_cache()
                                scene.recordpoints(iteration, "addionally prune_mask")
                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()

                        
            if densify == 7: # more general
                if iteration < opt.densify_until_iter :
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        if flag < opt.desicnt:
                            scene.recordpoints(iteration, "before densify")
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.densify_prunecloneimgeneral(opt.densify_grad_threshold, op.opthr, scene.cameras_extent, size_threshold)
                            flag+=1
                            scene.recordpoints(iteration, "after densify")
                        else:
                            prune_mask =  (gaussians.get_opacity < op.opthr).squeeze()
                            if opt.prunebysize :
                                big_points_vs = gaussians.max_radii2D > 20
                                big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                            gaussians.prune_points(prune_mask)
                            torch.cuda.empty_cache()
                            scene.recordpoints(iteration, "addionally prune_mask")
                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()


            if densify == 8: # more generate method also remove minmax points
                if iteration < opt.densify_until_iter :
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        if flag < opt.desicnt:
                            scene.recordpoints(iteration, "before densify")
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.densify_prunecloneimgeneral(opt.densify_grad_threshold, op.opthr, scene.cameras_extent, size_threshold)
                            flag+=1
                            scene.recordpoints(iteration, "after densify")
                        else:
                            prune_mask =  (gaussians.get_opacity < op.opthr).squeeze()
                            if opt.prunebysize :
                                big_points_vs = gaussians.max_radii2D > 20
                                big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                            gaussians.prune_points(prune_mask)
                            torch.cuda.empty_cache()
                            scene.recordpoints(iteration, "addionally prune_mask")
                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()

                if iteration == 10000: 
                    removeminmax(gaussians, maxbounds, minbounds)
                    torch.cuda.empty_cache()
                    scene.recordpoints(iteration, "addionally prune_mask")


           # after densification 
            if iteration > emsstartfromiterations and flagems == 2 and emscnt < selectedlength and viewpoint_cam.image_name in selectviews and (iteration - lasterems > 100): #
                scene.recordpoints(iteration, "current ems time " +  str(timeindex))
                

                selectviews.remove(viewpoint_cam.image_name)

                emscnt += 1
                lasterems = iteration

                diff = 1.0 - ssimmap(image.detach(), gt_image) # we choose ares with large d-ssim..
                diff = torch.sum(diff,        dim=0) # h, w
                diff_sorted, _ = torch.sort(diff.reshape(-1)) 
                numpixels = diff.shape[0] * diff.shape[1]
                threshold = diff_sorted[int(numpixels*opt.emsthr)].item()
                
                outmask = diff > threshold# 0.03  #error threshold
                kh, kw = 16, 16 # kernel size
                dh, dw = 16, 16 # stride
                idealh, idealw = int(image.shape[1] / dh  + 1) * kw, int(image.shape[2] / dw + 1) * kw # compute the ideal size for padding
                outmask = torch.nn.functional.pad(outmask, (0, idealw - outmask.shape[1], 0, idealh - outmask.shape[0]), mode='constant', value=0)

                patches = outmask.unfold(0, kh, dh).unfold(1, kw, dw)
                dummypatch = torch.ones_like(patches)
                patchessum = patches.sum(dim=(2,3)) 
                patchesmusk = patchessum  >  kh * kh * 0.85
                patchesmusk = patchesmusk.unsqueeze(2).unsqueeze(3).repeat(1,1,kh,kh).float()
                patches = dummypatch * patchesmusk

                # midpatch = torch.ones_like(patches)
                depth = render_pkg["depth"]
                depth = depth.squeeze(0)
                idealdepthh, idealdepthw = int(depth.shape[0] / dh  + 1) * kw, int(depth.shape[1] / dw + 1) * kw # compute the ideal size for padding

                depth = torch.nn.functional.pad(depth, (0, idealdepthw - depth.shape[1], 0, idealdepthh - depth.shape[0]), mode='constant', value=0)

                depthpaches = depth.unfold(0, kh, dh).unfold(1, kw, dw)
                dummydepthpatches =  torch.ones_like(depthpaches)
                a,b,c,d = depthpaches.shape
                depthpaches = depthpaches.reshape(a,b,c*d)
                mediandepthpatch = torch.median(depthpaches, dim=(2))[0]
                depthpaches = dummydepthpatches * (mediandepthpatch.unsqueeze(2).unsqueeze(3))
                unfold_depth_shape = dummydepthpatches.size()
                output_depth_h = unfold_depth_shape[0] * unfold_depth_shape[2]
                output_depth_w = unfold_depth_shape[1] * unfold_depth_shape[3]

                patches_depth_orig = depthpaches.view(unfold_depth_shape)
                patches_depth_orig = patches_depth_orig.permute(0, 2, 1, 3).contiguous()
                patches_depth = patches_depth_orig.view(output_depth_h, output_depth_w).float() # H * W  mask, # 1 for error, 0 for no error

                depth = patches_depth[:render_pkg["depth"].shape[1], :render_pkg["depth"].shape[2]]
                depth = depth.unsqueeze(0)

 
                midpatch = torch.ones_like(patches)

                centerpatches = patches * midpatch

                unfold_shape = patches.size()
                patches_orig = patches.view(unfold_shape)
                centerpatches_orig = centerpatches.view(unfold_shape)

                output_h = unfold_shape[0] * unfold_shape[2]
                output_w = unfold_shape[1] * unfold_shape[3]
                patches_orig = patches_orig.permute(0, 2, 1, 3).contiguous()
                centerpatches_orig = centerpatches_orig.permute(0, 2, 1, 3).contiguous()
                centermask = centerpatches_orig.view(output_h, output_w).float() # H * W  mask, # 1 for error, 0 for no error
                centermask = centermask[:image.shape[1], :image.shape[2]] # reverse back
                
                errormask = patches_orig.view(output_h, output_w).float() # H * W  mask, # 1 for error, 0 for no error
                errormask = errormask[:image.shape[1], :image.shape[2]] # reverse back

                H, W = centermask.shape

                offsetH = int(H/10)
                offsetW = int(W/10) # fish eye boundary artifacts, we don't sample there
                
                centermask[0:offsetH, :] = 0.0
                centermask[:, 0:offsetW] = 0.0

                centermask[-offsetH:, :] = 0.0
                centermask[:, -offsetW:] = 0.0


                depthmap = torch.cat((depth, depth, depth), dim=0)
                invaliddepthmask = depth == 15.0

                    


                pathdir = scene.model_path + "/ems_" + str(emscnt-1)
                if not os.path.exists(pathdir): 
                    os.makedirs(pathdir)
                
                depthmap = depthmap / torch.amax(depthmap)
                invalideptmap = torch.cat((invaliddepthmask, invaliddepthmask, invaliddepthmask), dim=0).float()  


                torchvision.utils.save_image(gt_image, os.path.join(pathdir,  "gt" + str(iteration) + ".png"))
                torchvision.utils.save_image(image, os.path.join(pathdir,  "render" + str(iteration) + ".png"))
                torchvision.utils.save_image(depthmap, os.path.join(pathdir,  "depth" + str(iteration) + ".png"))
                torchvision.utils.save_image(invalideptmap, os.path.join(pathdir,  "indepth" + str(iteration) + ".png"))
                  

                centermaskedimages =torch.stack((centermask, centermask, centermask), dim=2).float().cpu()  #0,1 
                centermaskedimages = centermaskedimages.numpy()
                # resize to x2 
                #maskedimages = cv2.resize()
                centermaskedimages = cv2.resize(centermaskedimages, dsize=(viewpoint_cam.image_width, viewpoint_cam.image_height), interpolation=cv2.INTER_CUBIC)




                # retrive current camera's K

              
                udcentermaskedimages = undistortimage(viewpoint_cam.image_name, dataset.source_path, centermaskedimages)
                gt_imagenumpy = gt_image.clone().permute(1,2,0).cpu().numpy()
                gt_imagex2 =  cv2.resize(gt_imagenumpy, dsize=(viewpoint_cam.image_width, viewpoint_cam.image_height), interpolation=cv2.INTER_CUBIC)
                gt_imagex2ud = undistortimage(viewpoint_cam.image_name, dataset.source_path, gt_imagex2)
                gt_imagex2udtorch = torch.from_numpy(gt_imagex2ud).cuda().permute(2,0,1)


           
                # use opencv undistort points to undistort these points
                udcentermaskedimages = np.sum(udcentermaskedimages, axis=2)
                udcentermaskedimages = torch.from_numpy(udcentermaskedimages).cuda()
                depthmask = udcentermaskedimages > torch.mean(udcentermaskedimages)# avoid close objects
                udcentermaskedimages = udcentermaskedimages * depthmask.float()

                undistortbadindics = (udcentermaskedimages > 1.0).nonzero() # baduvidx, viewpoint_cam, depthmap, gt_image, numperay=3
                
                

                #mediandepth = torch.median(depth)
                diff_sorted , _ = torch.sort(depth.reshape(-1)) 
                N = diff_sorted.shape[0]
                mediandepth = int(0.7 * N)
                mediandepth = diff_sorted[mediandepth]


                depth = torch.where(depth>mediandepth, depth,mediandepth )
                if opt.shuffleems == 0:
                    totalNnewpoints = gaussians.addgaussians(undistortbadindics, viewpoint_cam, depth, gt_imagex2udtorch.squeeze(0), numperay=opt.farray, ratioend=opt.rayends, depthmax=depthdict[viewpoint_cam.image_name])
                else:
                    totalNnewpoints = gaussians.addgaussians(undistortbadindics, viewpoint_cam, depth, gt_imagex2udtorch.squeeze(0), numperay=opt.farray, ratioend=opt.rayends, depthmax=depthdict[viewpoint_cam.image_name], shuffle=True)

                scene.recordpoints(iteration, "depth" + str(torch.max(depth).item()))



                udcentermaskedimagesbinary = udcentermaskedimages > 1.0 
                udcentermaskedimagesbinary = udcentermaskedimagesbinary.float()
                gt_image = gt_imagex2udtorch * udcentermaskedimagesbinary 
                image = render_pkg["render"] * errormask

                scene.recordpoints(iteration, "after addpointsbyuv" + viewpoint_cam.image_name)

       
                torchvision.utils.save_image(udcentermaskedimages, os.path.join(pathdir,  "maskedundistedmask" + str(iteration) + ".png"))
    
                torchvision.utils.save_image(gt_image, os.path.join(pathdir,  "maskedudgt" + str(iteration) + ".png"))
                torchvision.utils.save_image(image, os.path.join(pathdir,  "maskedrender" + str(iteration) + ".png"))
                visibility_filter = torch.cat((visibility_filter, torch.zeros(totalNnewpoints).cuda(0)), dim=0)
                visibility_filter = visibility_filter.bool()
                radii = torch.cat((radii, torch.zeros(totalNnewpoints).cuda(0)), dim=0)
                viewspace_point_tensor = torch.cat((viewspace_point_tensor, torch.zeros(totalNnewpoints, 3).cuda(0)), dim=0)


            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)





def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser; 
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser) #we put more parameters in optimization params, just for convenience.
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6029)
    parser.add_argument('--debug_from', type=int, default=-2)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 7_000,  10000, 12000, 15000, 20_000, 25_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[ 20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--densify", type=int, default=1, help="densify =1, we control points on N3d dataset")
    parser.add_argument("--duration", type=int, default=50, help="5 debug , 50 used")
    parser.add_argument("--basicfunction", type=str, default = "gaussian")
    parser.add_argument("--rgbfunction", type=str, default = "rgbv1")
    parser.add_argument("--rdpip", type=str, default = "v2")
    parser.add_argument("--configpath", type=str, default = "None")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet) # important !!!! seed 0, 


    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # incase we provide config file not directly pass to the file     config will overwrite the argument... to change to the reverse?
    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k) 
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
    else:
        print("config file not exist or not provided")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # refactor the code may affect results? unsure. keep the original structure
    args.iterations = 20000 # hard coded do not change

    train(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, densify=args.densify, duration=args.duration, basicfunction=args.basicfunction, rgbfunction=args.rgbfunction, rdpip=args.rdpip)

    # All done
    print("\nTraining complete.")
