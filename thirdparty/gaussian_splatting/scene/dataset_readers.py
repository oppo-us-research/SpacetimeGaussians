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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
import glob
import natsort
from simple_knn._C import distCUDA2
import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    near: float
    far: float
    timestamp: float
    pose: np.array 
    hpdirecitons: np.array
    cxr: float
    cyr: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50):
    cam_infos = []

    # pose in llff. pipeline by hypereel 
    originnumpy = os.path.join(os.path.dirname(os.path.dirname(images_folder)), "poses_bounds.npy")
    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)


        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        bounds = poses_bounds[:, -2:]


        near = bounds.min() * 0.95
        far = bounds.max() * 1.05
        
        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # 19, 3, 5





        H, W, focal = poses[0, :, -1]
        cx, cy = W / 2.0, H / 2.0

        K = np.eye(3)
        K[0, 0] = focal * W / W / 2.0
        K[0, 2] = cx * W / W / 2.0
        K[1, 1] = focal * H / H / 2.0
        K[1, 2] = cy * H / H / 2.0
        
        imageH = int (H//2) # note hard coded to half of the original image size
        imageW = int (W//2)
      
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)
    
    sortedtotalcamelist =  natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in  range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i # map each cam with a number
     

    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]




        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"



       
        for j in range(startime, startime+ int(duration)):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image_path = image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            image = Image.open(image_path)
            if j == startime:
                # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=hpposes[sortednamedict[os.path.basename(extr.name)]], hpdirecitons=hpdirecitons,cxr=0.0, cyr=0.0)
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=1,cxr=0.0, cyr=0.0)

            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None, cxr=0.0, cyr=0.0)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasTechnicolor(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50):

    cam_infos = []
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)
    
    sortedtotalcamelist =  natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in  range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i # map each cam with a number

    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"



       
        for j in range(startime, startime+ int(duration)):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image_path = image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)


        
            cxr =   ((intr.params[2] )/  width - 0.5) 
            cyr =   ((intr.params[3] ) / height - 0.5) 

            K = np.eye(3)
            K[0, 0] = focal_length_x #* 0.5
            K[0, 2] = intr.params[2] #* 0.5 
            K[1, 1] = focal_length_y #* 0.5
            K[1, 2] = intr.params[3] #* 0.5



            halfH = round(height / 2.0 )
            halfW = round(width / 2.0 )
            
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)

            
            image = Image.open(image_path)


            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr)
            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None,  cxr=cxr, cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos



def normalize(v):
    return v / np.linalg.norm(v)


def readColmapCamerasMv(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50):
    cam_infos = []
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCV


    for idx, key in enumerate(cam_extrinsics): 
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()

        cxr =   ((intr.params[2] )/  width - 0.5) 
        cyr =   ((intr.params[3] ) / height - 0.5) 
 
        if extr.name == "cam00.png":
            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            if cyr != 0.0 :
                cxr = cxr
                cyr = cyr
                projection_matrix = getProjectionMatrixCV(znear=0.01, zfar=100.0, fovX=FovX, fovY=FovY, cx=cxr, cy=cyr).transpose(0,1).cuda()
            else:
                projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FovX, fovY=FovY).transpose(0,1).cuda()

            camera_center = world_view_transform.inverse()[3, :3]

            projectinverse = projection_matrix.T.inverse()
            camera2wold = world_view_transform.T.inverse()
            
            ndccamera = torch.Tensor((0, 0, 1, 1)).cuda()
            ndccamera = ndccamera.unsqueeze(0) # 1, 4
            projected = ndccamera @ projectinverse.T # 1, 4,  @ 4,4 
            diretioninlocal = projected / projected[:,3:] #v 
            direction = diretioninlocal[:,:3] @ camera2wold[:3,:3].T 
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)  

            target = camera_center + rays_d * 30.0
            break 

        
    radiace = 1.0

    for i in range(240):
        theta = i / 240.0 * 4 * np.pi
        newcameracenter = camera_center + radiace * torch.Tensor((np.cos(theta), np.sin(theta), 2.0 + 2.0*np.sin(theta))).cuda()
        
        newforward_vector = target - newcameracenter
        newforward_vector = newforward_vector.cpu().numpy()

        right_vector = R[:, 0]  # First column
        up_vector = R[:, 1]    # Second column
        forward_vector = R[:, 2]  # Third column

        newright = normalize(np.cross(up_vector, newforward_vector))
        up = normalize(np.cross(newforward_vector, newright))

        newR = np.eye(3)
        newR[:, 0] = newright 
        newR[:, 1] = up
        newR[:, 2] = normalize(newforward_vector) 


        C2W = np.zeros((4, 4))
        C2W[:3, :3] = newR
        C2W[:3, 3] = newcameracenter.cpu().numpy()
        C2W[3, 3] = 1.0
        rt = np.linalg.inv(C2W)  
        newt = rt[:3, 3]
        
        image_name = "mv_" + str(i)
        uid = i

        time = (i)/240
        cam_info = CameraInfo(uid=uid, R=newR, T=newt, FovY=FovY, FovX=FovX, image=None, image_path=None, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=time, pose=1, hpdirecitons=0, cxr=0.0, cyr=0.0)
        
        cam_infos.append(cam_info)
        
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasImmersive(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50):
    cam_infos = []

      
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)
    
    sortedtotalcamelist =  natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in  range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i # map each cam with a number

    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        # if extr.name not in ["camera_0005.png", "camera_0001.png"]:
        #         continue 
       
        for j in range(startime, startime+ int(duration)):
            # image_path = os.path.join(images_folder, os.path.basename(extr.name))
            # image_name = os.path.basename(image_path).split(".")[0]
            # image_path = image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)

            parentfolder = os.path.dirname(images_folder)
            parentfolder = os.path.dirname(parentfolder)
            image_name = extr.name.split(".")[0]

            rawvideofolder = os.path.join(parentfolder,os.path.basename(image_name))

            image_path = os.path.join(rawvideofolder, str(j) + ".png")
            
            #image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)
            
            # K = np.eye(3)
            # K[0, 0] = focal_length_x * 0.5
            # K[0, 2] = intr.params[2] * 0.5 
            # K[1, 1] = focal_length_y * 0.5
            # K[1, 2] = intr.params[3] * 0.5


        
            cxr =   ((intr.params[2] )/  width - 0.5) 
            cyr =   ((intr.params[3] ) / height - 0.5) 

            K = np.eye(3)
            K[0, 0] = focal_length_x #* 0.5
            K[0, 2] = intr.params[2] #* 0.5 
            K[1, 1] = focal_length_y #* 0.5
            K[1, 2] = intr.params[3] #* 0.5


            if not os.path.exists(image_path):
                image_path = image_path.replace("_S14","")
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            image = Image.open(image_path)
            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr)
            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None,  cxr=cxr, cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos





def readColmapCamerasImmersiveTestonly(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50):
    cam_infos = []

      
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)
    
    sortedtotalcamelist =  natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in  range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i # map each cam with a number

    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        # T[1] = T[1] + 0.2  # y incrase by 1
        # T[2] = T[2] + 0.65 
        # T[0] = T[0] + 0.65 # x by 0.65
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"



       
        for j in range(startime, startime+ int(duration)):
            # image_path = os.path.join(images_folder, os.path.basename(extr.name))
            # image_name = os.path.basename(image_path).split(".")[0]
            # image_path = image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)

            parentfolder = os.path.dirname(images_folder)
            parentfolder = os.path.dirname(parentfolder)
            image_name = extr.name.split(".")[0]

            rawvideofolder = os.path.join(parentfolder,os.path.basename(image_name))

            image_path = os.path.join(rawvideofolder, str(j) + ".png")
            
            #image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)
            
            # K = np.eye(3)
            # K[0, 0] = focal_length_x * 0.5
            # K[0, 2] = intr.params[2] * 0.5 
            # K[1, 1] = focal_length_y * 0.5
            # K[1, 2] = intr.params[3] * 0.5


        
            cxr =   ((intr.params[2] )/  width - 0.5) 
            cyr =   ((intr.params[3] ) / height - 0.5) 

            K = np.eye(3)
            K[0, 0] = focal_length_x #* 0.5
            K[0, 2] = intr.params[2] #* 0.5 
            K[1, 1] = focal_length_y #* 0.5
            K[1, 2] = intr.params[3] #* 0.5



            #halfH = round(height / 2.0 )
            #halfW = round(width / 2.0 )
            if not os.path.exists(image_path):
                image_path = image_path.replace("_S14","")
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)

            if image_name == "camera_0001":
                image = Image.open(image_path)
            else:
                image = None 
            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr)
            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None,  cxr=cxr, cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    times = np.vstack([vertices['t']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)

def storePly(path, xyzt, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('t','f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    xyz = xyzt[:, :3]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyzt.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzt, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfoImmersive(path, images, eval, llffhold=8, multiview=False, duration=50, testonly=False ):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    
    near = 0.01
    far = 100

    starttime = os.path.basename(path).split("_")[1] # colmap_0, 
    assert starttime.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"
    starttime = int(starttime)
    
    # readColmapCamerasImmersiveTestonly
    if testonly:
        cam_infos_unsorted = readColmapCamerasImmersiveTestonly(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), near=near, far=far, startime=starttime, duration=duration)
    else:
        cam_infos_unsorted = readColmapCamerasImmersive(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), near=near, far=far, startime=starttime, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
     

    if eval:
        train_cam_infos =  cam_infos[duration:]  # + cam_infos[:duration] # for demo only
        test_cam_infos = cam_infos[:duration]
        uniquecheck = []
        for cam_info in test_cam_infos:
            if cam_info.image_name not in uniquecheck:
                uniquecheck.append(cam_info.image_name)
        assert len(uniquecheck) == 1 
        
        sanitycheck = []
        for cam_info in train_cam_infos:
            if cam_info.image_name not in sanitycheck:
                sanitycheck.append(cam_info.image_name)
        for testname in uniquecheck:
            assert testname not in sanitycheck
    else:  
        train_cam_infos = cam_infos # for demo without eval
        test_cam_infos = cam_infos[:duration]




    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    totalply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")
    
    # if os.path.exists(ply_path):
    #     os.remove(ply_path)
    
    
    # if os.path.exists(totalply_path):
    #     os.remove(totalply_path)
    
    if not os.path.exists(totalply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        totalxyz = []
        totalrgb = []
        totaltime = []

        takeoffset = 0
        for i in range(starttime, starttime + duration):
            thisbin_path = os.path.join(path, "sparse/0/points3D.bin").replace("colmap_"+ str(starttime), "colmap_" + str(i), 1)
            xyz, rgb, _ = read_points3D_binary(thisbin_path)
            
            totalxyz.append(xyz)
            totalrgb.append(rgb)
            totaltime.append(np.ones((xyz.shape[0], 1)) * (i-starttime) / duration)
        xyz = np.concatenate(totalxyz, axis=0)
        rgb = np.concatenate(totalrgb, axis=0)
        totaltime = np.concatenate(totaltime, axis=0)
        assert xyz.shape[0] == rgb.shape[0]  
        xyzt =np.concatenate( (xyz, totaltime), axis=1)     
        storePly(totalply_path, xyzt, rgb)
    try:
        pcd = fetchPly(totalply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=totalply_path)
    return scene_info




def readColmapSceneInfoMv(path, images, eval, llffhold=8, multiview=False, duration=50):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    parentdir = os.path.dirname(path)

    near = 0.01
    far = 100

    starttime = os.path.basename(path).split("_")[1] # colmap_0, 
    assert starttime.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"
    starttime = int(starttime)
    

    cam_infos_unsorted = readColmapCamerasMv(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), near=near, far=far, startime=starttime, duration=duration)
    cam_infos = cam_infos_unsorted
     
    # for cam in cam_infos:
    #     print(cam.image_name)
    # for cam_info in cam_infos:
    #     print(cam_info.uid, cam_info.R, cam_info.T, cam_info.FovY, cam_info.image_name)
    
    train_cam_infos = []
    test_cam_infos = cam_infos


    nerf_normalization = getNerfppNorm(test_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    totalply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + "_mv.ply")
    
    # if os.path.exists(ply_path):
    #     os.remove(ply_path)
    
    
    # if os.path.exists(totalply_path):
    #     os.remove(totalply_path)
    
    # # if not os.path.exists(totalply_path):
    # #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    # #     totalxyz = []
    # #     totalrgb = []
    # #     totaltime = []
    # #     for i in range(starttime, starttime + duration):
    # #         thisbin_path = os.path.join(path, "sparse/0/points3D.bin").replace("colmap_"+ str(starttime), "colmap_" + str(i), 1)
    # #         xyz, rgb, _ = read_points3D_binary(thisbin_path)
    # #         totalxyz.append(xyz)
    # #         totalrgb.append(rgb)
    # #         totaltime.append(np.ones((xyz.shape[0], 1)) * (i-starttime) / duration)
    # #     xyz = np.concatenate(totalxyz, axis=0)
    # #     rgb = np.concatenate(totalrgb, axis=0)
    # #     totaltime = np.concatenate(totaltime, axis=0)
    # #     assert xyz.shape[0] == rgb.shape[0]  
    # #     xyzt =np.concatenate( (xyz, totaltime), axis=1)     
    # #     storePly(totalply_path, xyzt, rgb)
    # # try:
    # #     pcd = fetchPly(totalply_path)
    # # except:
    pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=totalply_path)
    return scene_info



def readColmapSceneInfo(path, images, eval, llffhold=8, multiview=False, duration=50):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    parentdir = os.path.dirname(path)

    near = 0.01
    far = 100

    starttime = os.path.basename(path).split("_")[1] # colmap_0, 
    assert starttime.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"
    starttime = int(starttime)
    

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), near=near, far=far, startime=starttime, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
     

    if eval:
        train_cam_infos =  cam_infos[duration:] 
        test_cam_infos = cam_infos[:duration]
        uniquecheck = []
        for cam_info in test_cam_infos:
            if cam_info.image_name not in uniquecheck:
                uniquecheck.append(cam_info.image_name)
        assert len(uniquecheck) == 1 
        
        sanitycheck = []
        for cam_info in train_cam_infos:
            if cam_info.image_name not in sanitycheck:
                sanitycheck.append(cam_info.image_name)
        for testname in uniquecheck:
            assert testname not in sanitycheck
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos[:2] #dummy

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    totalply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")
    

    
    if not os.path.exists(totalply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        totalxyz = []
        totalrgb = []
        totaltime = []
        for i in range(starttime, starttime + duration):
            thisbin_path = os.path.join(path, "sparse/0/points3D.bin").replace("colmap_"+ str(starttime), "colmap_" + str(i), 1)
            xyz, rgb, _ = read_points3D_binary(thisbin_path)
            totalxyz.append(xyz)
            totalrgb.append(rgb)
            totaltime.append(np.ones((xyz.shape[0], 1)) * (i-starttime) / duration)
        xyz = np.concatenate(totalxyz, axis=0)
        rgb = np.concatenate(totalrgb, axis=0)
        totaltime = np.concatenate(totaltime, axis=0)
        assert xyz.shape[0] == rgb.shape[0]  
        xyzt =np.concatenate( (xyz, totaltime), axis=1)     
        storePly(totalply_path, xyzt, rgb)
    try:
        pcd = fetchPly(totalply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=totalply_path)
    return scene_info



def readColmapSceneInfoTechnicolor(path, images, eval, llffhold=8, multiview=False, duration=50):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    parentdir = os.path.dirname(path)


    starttime = os.path.basename(path).split("_")[1] # colmap_0, 
    assert starttime.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"
    starttime = int(starttime)
    
    near = 0.01
    far = 100
    cam_infos_unsorted = readColmapCamerasTechnicolor(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), near=near, far=far, startime=starttime, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
     
    # for cam in cam_infos:
    #     print(cam.image_name)
    # for cam_info in cam_infos:
    #     print(cam_info.uid, cam_info.R, cam_info.T, cam_info.FovY, cam_info.image_name)


    if eval:
            train_cam_infos = [_ for _ in cam_infos if "cam10" not in _.image_name]
            test_cam_infos = [_ for _ in cam_infos if "cam10" in _.image_name]
            if len(test_cam_infos) > 0:
                uniquecheck = []
                for cam_info in test_cam_infos:
                    if cam_info.image_name not in uniquecheck:
                        uniquecheck.append(cam_info.image_name)
                assert len(uniquecheck) == 1 
                
                sanitycheck = []
                for cam_info in train_cam_infos:
                    if cam_info.image_name not in sanitycheck:
                        sanitycheck.append(cam_info.image_name)
                for testname in uniquecheck:
                    assert testname not in sanitycheck
            else:
                first_cam = cam_infos[0].image_name
                print("do custom loader training, select first cam as test frame: ", first_cam)
                cam_infos = natsort.natsorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
                train_cam_infos = [_ for _ in cam_infos if first_cam not in _.image_name]
                test_cam_infos = [_ for _ in cam_infos if first_cam in _.image_name]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos[:4]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    totalply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")
    

    if not os.path.exists(totalply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        totalxyz = []
        totalrgb = []
        totaltime = []
        for i in range(starttime, starttime + duration):
            thisbin_path = os.path.join(path, "sparse/0/points3D.bin").replace("colmap_"+ str(starttime), "colmap_" + str(i), 1)
            xyz, rgb, _ = read_points3D_binary(thisbin_path)
            totalxyz.append(xyz)
            totalrgb.append(rgb)
            totaltime.append(np.ones((xyz.shape[0], 1)) * (i-starttime) / duration)
        xyz = np.concatenate(totalxyz, axis=0)
        rgb = np.concatenate(totalrgb, axis=0)
        totaltime = np.concatenate(totaltime, axis=0)
        assert xyz.shape[0] == rgb.shape[0]  
        xyzt =np.concatenate( (xyz, totaltime), axis=1)     
        storePly(totalply_path, xyzt, rgb)
    try:
        pcd = fetchPly(totalply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=totalply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            
            for j in range(20):
                cam_infos.append(CameraInfo(uid=idx*20 + j, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", multiview=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info









def readColmapCamerasImmersivev2Testonly(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50):
    cam_infos = []

      
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)
    
    sortedtotalcamelist =  natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in  range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i # map each cam with a number

    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"



       
        for j in range(startime, startime+ int(duration)):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image_path = image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)

            # parentfolder = os.path.dirname(images_folder)
            # parentfolder = os.path.dirname(parentfolder)
            # image_name = extr.name.split(".")[0]

            #rawvideofolder = os.path.join(parentfolder,os.path.basename(image_name))

            #image_path = os.path.join(rawvideofolder, str(j) + ".png")


            cxr =   ((intr.params[2] )/  width - 0.5) 
            cyr =   ((intr.params[3] ) / height - 0.5) 

            K = np.eye(3)
            K[0, 0] = focal_length_x #* 0.5
            K[0, 2] = intr.params[2] #* 0.5 
            K[1, 1] = focal_length_y #* 0.5
            K[1, 2] = intr.params[3] #* 0.5



            #halfH = round(height / 2.0 )
            #halfW = round(width / 2.0 )
            
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)

            if image_name == "camera_0001":
                image = Image.open(image_path)
            else:
                image = None 
            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr)
            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None,  cxr=cxr, cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos



def readColmapCamerasImmersivev2(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50):
    cam_infos = []

      
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)
    
    sortedtotalcamelist =  natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in  range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i # map each cam with a number

    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"



       
        for j in range(startime, startime+ int(duration)):
            # image_path = os.path.join(images_folder, os.path.basename(extr.name))
            # image_name = os.path.basename(image_path).split(".")[0]
            # image_path = image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)

            parentfolder = os.path.dirname(images_folder)
            parentfolder = os.path.dirname(parentfolder)
            image_name = extr.name.split(".")[0]

            rawvideofolder = os.path.join(parentfolder,os.path.basename(image_name))

            image_path = os.path.join(rawvideofolder, str(j) + ".png")
            
            #image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)
            
            # K = np.eye(3)
            # K[0, 0] = focal_length_x * 0.5
            # K[0, 2] = intr.params[2] * 0.5 
            # K[1, 1] = focal_length_y * 0.5
            # K[1, 2] = intr.params[3] * 0.5


        
            cxr =   ((intr.params[2] )/  width - 0.5) 
            cyr =   ((intr.params[3] ) / height - 0.5) 




            hpdirecitons = 1.0
            
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            image = Image.open(image_path)
            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=hpdirecitons, cxr=cxr, cyr=cyr)
            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None,  cxr=cxr, cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos
    
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Immersive": readColmapSceneInfoImmersive,
    "Colmapmv": readColmapSceneInfoMv,
    "Blender" : readNerfSyntheticInfo, 
    "Technicolor": readColmapSceneInfoTechnicolor,
}


