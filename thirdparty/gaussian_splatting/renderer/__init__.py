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

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE ####################################
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################




import torch
import math
import time 
import torch.nn.functional as F
import time 




from scene.oursfull import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getProjectionMatrixCV, focal2fov, fov2focal




def train_ours_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3
    rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth}




def test_ours_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    torch.cuda.synchronize()
    startime = time.time()

    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3
    rendered_image = rendered_image.squeeze(0)
    torch.cuda.synchronize()
    duration = time.time() - startime 

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration":duration}


def test_ours_full_fused(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):

 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
     
    try: 
        getattr(pc.rgbdecoder, "sigmoid")
        usesigmoid = True 
    except:
        usesigmoid = False 

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0


    torch.cuda.synchronize()
    startime = time.time()#perf_counter

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False)

    rasterizer = GRzer(raster_settings=raster_settings)
    
    


    tforpoly = viewpoint_camera.timestamp - pc.get_trbfcenter
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    
    means2D = screenspace_points
   

    cov3D_precomp = None

    #scales = pc.get_scaling

    shs = None
    # sandwichold 2,3
    rendered_image, radii = rasterizer(
        timestamp = viewpoint_camera.timestamp, 
        trbfcenter = pc.get_trbfcenter,
        trbfscale = pc.computedtrbfscale ,
        motion = pc._motion,
        means3D = pc.get_xyz,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = pc.computedopacity,
        scales = pc.computedscales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        mlp1 = pc.rgbdecoder.mlp1.weight.squeeze(3).squeeze(2).flatten(), 
        mlp2 = pc.rgbdecoder.mlp2.weight.squeeze(3).squeeze(2).flatten(),
        rayimage = viewpoint_camera.rays)
    # add sigmoid for n3d dataset, but should be even faster in the cuda   
    if usesigmoid == True:
        rendered_image = torch.sigmoid(rendered_image)
    else:
        pass 
    torch.cuda.synchronize()
    duration = time.time() - startime #

    # INTER_CUBIC
    rendered_image = rendered_image.squeeze(0)


    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "rawrendered_image": rendered_image,
             "duration": duration}



def test_ours_lite(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None,GRsetting=None, GRzer=None):

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    torch.cuda.synchronize()
    startime = time.time()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False)

    rasterizer = GRzer(raster_settings=raster_settings)
    
    


    tforpoly = viewpoint_camera.timestamp - pc.get_trbfcenter
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    
    means2D = screenspace_points
   

    cov3D_precomp = None


    shs = None
 
    rendered_image, radii = rasterizer(
        timestamp = viewpoint_camera.timestamp, 
        trbfcenter = pc.get_trbfcenter,
        trbfscale = pc.computedtrbfscale ,
        motion = pc._motion,
        means3D = pc.get_xyz,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = pc.computedopacity,
        scales = pc.computedscales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    
    torch.cuda.synchronize()
    duration = time.time() - startime 
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "duration":duration}




def train_ours_lite(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly

    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth}


def train_ours_fullss(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    opacity = pointopacity * trbfoutput  # - 0.5

    pc.trbfoutput = trbfoutput
    cov3D_precomp = None

    scales = pc.get_scaling

    shs = None
    
    tforpoly = trbfdistanceoffset.detach()

    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly

    rotations = pc.get_rotation(tforpoly)


    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rawrendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays) # 1 , 3      
    newiamge = F.grid_sample(rawrendered_image, viewpoint_camera.fisheyemapper, mode='bicubic', padding_mode='zeros', align_corners=True)
    # INTER_CUBIC
    rendered_image = F.interpolate(newiamge, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
    #rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)# 
  
    rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "rawrendered_image": rawrendered_image,
            "depth": depth}




def test_ours_fullss_fused(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    pc.rgbdecoder.mlp2.weight.shape (3,6,1,1)

    pc.rgbdecoder.mlp2.weight.flatten()[0:6] = pc.rgbdecoder.mlp2.weight[0]

    # https://stackoverflow.com/questions/76949152/why-does-pytorchs-linear-layer-store-the-weight-in-shape-out-in-and-transpos
    # store in (outfeature, infeature) order..
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # mapperss = viewpoint_camera.fisheyemapper.permute(0,3,1,2) 
    # mapperoriginsize =  F.interpolate(mapperss, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
    # mapperoriginsize = mapperoriginsize.permute(0,2,3,1)


    torch.cuda.synchronize()
    startime = time.time()#perf_counter

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False)

    rasterizer = GRzer(raster_settings=raster_settings)
    
    


    tforpoly = viewpoint_camera.timestamp - pc.get_trbfcenter
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    
    means2D = screenspace_points
   

    cov3D_precomp = None

    #scales = pc.get_scaling

    shs = None
    # sandwichold 2,3
    rendered_image, radii = rasterizer(
        timestamp = viewpoint_camera.timestamp, 
        trbfcenter = pc.get_trbfcenter,
        trbfscale = pc.computedtrbfscale ,
        motion = pc._motion,
        means3D = pc.get_xyz,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = pc.computedopacity,
        scales = pc.computedscales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        mlp1 = pc.rgbdecoder.mlp1.weight.squeeze(3).squeeze(2).flatten(),
        mlp2 = pc.rgbdecoder.mlp2.weight.squeeze(3).squeeze(2).flatten(),
        rayimage = viewpoint_camera.rays)

    torch.cuda.synchronize()
    duration = time.time() - startime #

    rendered_image = F.grid_sample(rendered_image.unsqueeze(0), viewpoint_camera.fisheyemapper, mode='bicubic', padding_mode='zeros', align_corners=True)
    # INTER_CUBIC
    rendered_image = F.interpolate(rendered_image, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
    rendered_image = rendered_image.squeeze(0)


    # rendered_image = F.grid_sample(rendered_image.unsqueeze(0), mapperoriginsize, mode='bicubic', padding_mode='zeros', align_corners=True)
    #  rawrendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays) # 1 , 3      
    ## rendered_image = rendered_image #[0:3,:,:]
    # rendered_image = torch.sigmoid(rendered_image) # sigmoid not fused.
    #rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "rawrendered_image": rendered_image,
             "duration": duration}



def test_ours_fullss(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    torch.cuda.synchronize()
    startime = time.time()#perf_counter

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    opacity = pointopacity * trbfoutput  # - 0.5

    pc.trbfoutput = trbfoutput
    cov3D_precomp = None

    scales = pc.get_scaling

    shs = None
    
    tforpoly = trbfdistanceoffset.detach()

    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly

    rotations = pc.get_rotation(tforpoly)


    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rawrendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays) # 1 , 3      
    newiamge = F.grid_sample(rawrendered_image, viewpoint_camera.fisheyemapper, mode='bicubic', padding_mode='zeros', align_corners=True)
    # INTER_CUBIC
    rendered_image = F.interpolate(newiamge, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)

    torch.cuda.synchronize()
    duration = time.time() - startime
    
    rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "rawrendered_image": rawrendered_image,
            "depth": depth,
            "duration":duration}







def train_ours_litess(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly

    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rawrendered_image = rendered_image.unsqueeze(0)  
    newiamge = F.grid_sample(rawrendered_image, viewpoint_camera.fisheyemapper, mode='bicubic', padding_mode='zeros', align_corners=True)
    # INTER_CUBIC
    rendered_image = F.interpolate(newiamge, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
    #rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)# 
  
    rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "rawrendered_image": rawrendered_image,
            "depth": depth}




def test_ours_litess(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None,GRsetting=None, GRzer=None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0


    torch.cuda.synchronize()


    startime = time.time()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False)

    rasterizer = GRzer(raster_settings=raster_settings)    


    tforpoly = viewpoint_camera.timestamp - pc.get_trbfcenter
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    
    means2D = screenspace_points
   

    cov3D_precomp = None


    shs = None
 
    rendered_image, radii = rasterizer(
        timestamp = viewpoint_camera.timestamp, 
        trbfcenter = pc.get_trbfcenter,
        trbfscale = pc.computedtrbfscale ,
        motion = pc._motion,
        means3D = pc.get_xyz,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = pc.computedopacity,
        scales = pc.computedscales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    torch.cuda.synchronize()
    duration = time.time() - startime #

    rendered_image = F.grid_sample(rendered_image.unsqueeze(0), viewpoint_camera.fisheyemapper, mode='bicubic', padding_mode='zeros', align_corners=True)
    # INTER_CUBIC
    rendered_image = F.interpolate(rendered_image, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
    rendered_image = rendered_image.squeeze(0)


    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "duration":duration}
