#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.oursfull import GaussianModel
from arguments import ModelParams
from PIL import Image 
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfosv2, cameraList_from_camInfosv2nogt
from helper_train import recordpointshelper, getfisheyemapper
import torch 
class Scene:

    # gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0], multiview=False,duration=50.0, loader="colmap"):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.refmodelpath = None
    

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        raydict = {}


        if loader == "colmap" or loader == "colmapvalid": # colmapvalid only for testing
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, multiview, duration=duration)
        
        elif loader == "technicolor" or loader == "technicolorvalid" :
            scene_info = sceneLoadTypeCallbacks["Technicolor"](args.source_path, args.images, args.eval, multiview, duration=duration)
        
        elif loader == "immersive" or loader == "immersivevalid" or loader == "immersivess"  :
            scene_info = sceneLoadTypeCallbacks["Immersive"](args.source_path, args.images, args.eval, multiview, duration=duration)
        elif loader == "immersivevalidss":
            scene_info = sceneLoadTypeCallbacks["Immersive"](args.source_path, args.images, args.eval, multiview, duration=duration, testonly=True)

        elif loader == "colmapmv" : # colmapvalid only for testing
            scene_info = sceneLoadTypeCallbacks["Colmapmv"](args.source_path, args.images, args.eval, multiview, duration=duration)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=2)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

 


        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")  
            if loader in ["colmapvalid", "colmapmv", "immersivevalid","technicolorvalid", "immersivevalidss", "imv2valid"]:         
                self.train_cameras[resolution_scale] = [] # no training data


            elif loader in ["immersivess"]:
                assert resolution_scale == 1.0, "High frequency data only available at 1.0 scale"
                self.train_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.train_cameras, resolution_scale, args, ss=True)

            else: # immersive and immersivevalid 
                self.train_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.train_cameras, resolution_scale, args)
            
            
            
            print("Loading Test Cameras")
            if loader  in ["colmapvalid", "immersivevalid", "colmap", "technicolorvalid", "technicolor", "imv2","imv2valid"]: # we need gt for metrics
                self.test_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.test_cameras, resolution_scale, args)
            elif loader in ["immersivess", "immersivevalidss"]:
                self.test_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.test_cameras, resolution_scale, args, ss=True)
            elif loader in ["colmapmv"]:                 # only for multi view

                self.test_cameras[resolution_scale] = cameraList_from_camInfosv2nogt(scene_info.test_cameras, resolution_scale, args)


        for cam in self.train_cameras[resolution_scale]:
            if cam.image_name not in raydict and cam.rayo is not None:
                # rays_o, rays_d = 1, cameradirect
                
                raydict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda() # 1 x 6 x H x W
        for cam in self.test_cameras[resolution_scale]:
            if cam.image_name not in raydict and cam.rayo is not None:
                raydict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda() # 1 x 6 x H x W

        for cam in self.train_cameras[resolution_scale]:
            cam.rays = raydict[cam.image_name] # should be direct ?

        for cam in self.test_cameras[resolution_scale]:
            cam.rays = raydict[cam.image_name] # should be direct ?

        if loader in ["immersivess", "immersivevalidss"]:# construct shared fisheyd remapping
            self.fisheyemapper = {}
            for cam in self.train_cameras[resolution_scale]:
                if cam.image_name not in self.fisheyemapper:
                    self.fisheyemapper[cam.image_name] = getfisheyemapper(args.source_path, cam.image_name) # 
                    self.fisheyemapper[cam.image_name].requires_grad = False
    
            for cam in self.test_cameras[resolution_scale]:
                if cam.image_name not in self.fisheyemapper:
                    self.fisheyemapper[cam.image_name] = getfisheyemapper(args.source_path, cam.image_name) # 
                    self.fisheyemapper[cam.image_name].requires_grad = False
            
            for cam in self.train_cameras[resolution_scale]:
                cam.fisheyemapper = self.fisheyemapper[cam.image_name]
            for cam in self.test_cameras[resolution_scale]:
                cam.fisheyemapper = self.fisheyemapper[cam.image_name]

       
        if self.loaded_iter :
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    # recordpointshelper(model_path, numpoints, iteration, string):
    def recordpoints(self, iteration, string):
        txtpath = os.path.join(self.model_path, "exp_log.txt")
        numpoints = self.gaussians._xyz.shape[0]
        recordpointshelper(self.model_path, numpoints, iteration, string)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
 