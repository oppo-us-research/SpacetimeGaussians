#!/bin/bash


conda create -n feature_splatting python=3.7.13
conda activate feature_splatting

# seems that we sometimes got stuck in environment.yml, so we install the packages one by one
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge


# Install for Gaussian Rasterization (Ch9) - Ours-Full
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9

# Install for Gaussian Rasterization (Ch3) - Ours-Lite
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch3

# Install for Forward Full - Ours-Full (speed up testing, mlp fused, no sigmoid)
pip install thirdparty/gaussian_splatting/submodules/forward_full

# Install for Forward Lite - Ours-Lite (speed up testing)
pip install thirdparty/gaussian_splatting/submodules/forward_lite


# install simpleknn
pip install thirdparty/gaussian_splatting/submodules/simple-knn

# install opencv-python-headless, to work with colmap on server
pip install opencv-python
# Install MMCV for CUDA KNN, used for init point sampling, reduce number of points when sfm points are too many
pip install -e thirdparty/mmcv -v # take ~30min; if mmcv dir is empty: `git submodule update --init` (or git clone with --recursive)

# other packages
pip install natsort
pip install scipy
pip install kornia
# install colmap for preprocess, work with python3.8
conda create -n colmapenv python=3.8
conda activate colmapenv
pip install opencv-python-headless
pip install tqdm
pip install natsort
pip install Pillow
# just some files need torch be installed.
conda install pytorch==1.12.1 -c pytorch -c conda-forge
conda config --set channel_priority false
conda install colmap -c conda-forge

conda activate feature_splatting




## Command to build real-time demo (optianal, Windows only, inference only, we provide the pre-built demo)
# ```
# cd thirdparty
# git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
# cd gaussian-splatting
# cd SIBR_viewers
# cmake -Bbuild .
# ```
# after ```cmake -Bbuild .```, it will automaticly donwload third party libraries provided by Inria (takes several minutes for the first time). </br>
# We need to update two default folders (```CudaRasterizer```, ```projects```) with our provided codes.  
# Just, update everthing except .git in ```extlibs\CudaRasterizer``` with our ```thirdparty\gaussian_splatting\realtimedemolite\extlibs\CudaRasterizer```
# And, update everthing except .git in  ```src\projects``` with our ```thirdparty\gaussian_splatting\realtimedemolite\projects```
# during replaceing, please keep the orginal .git dirtory. only replace the code
# after manually update the two folders, you can rebuild the projects
# ```
# cmake -Bbuild .
# cmake --build build --target install --config RelWithDebInfo
# ```


##### 
# 1. do not remove orignal .git repo in the extlib/cudarasterizor (as the cmake function will use .git to check the exisiting of folder during building, otherwise a new version of original cudarasterizer is downloaded), please just replace the source file only

# 2. if you meet cuda not found error and driver is cuda12 and cuda-toolkit(nvcc 11.8) on windows with visual studio 

# # # remove CUDA 12.xxx in C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations  if you see  11.8

# # # building process will select the highest cuda version, so delete CUDA 12.xxx if you want to build with CUDA nvcc 11.8 

#### reactivate conda environment if cuda extension is built ? (not sure)
