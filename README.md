# Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis

[Project Page](https://oppo-us-research.github.io/SpacetimeGaussians-website/) | [Paper](https://arxiv.org/abs/2312.16812) | [Video](https://youtu.be/YsPPmf-E6Lg) | [Viewer & Pre-Trained Models](https://huggingface.co/stack93/spacetimegaussians/tree/main)


This is an official implementation of the paper "Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis".</br>
[Zhan Li](https://lizhan17.github.io/web/)<sup>1,2</sup>, 
[Zhang Chen](https://zhangchen8.github.io/)<sup>1,&dagger;</sup>, 
[Zhong Li](https://sites.google.com/site/lizhong19900216)<sup>1,&dagger;</sup>, 
[Yi Xu](https://www.linkedin.com/in/yi-xu-42654823/)<sup>1</sup> </br>
<sup>1</sup> OPPO US Research Center, <sup>2</sup> Portland State University </br>
<sup>&dagger;</sup> Corresponding authors </br>

<img src="assets/output.gif" width="100%"/></br>

## Table of Contents
1. [Installation](#installation)
1. [Preprocess Datasets](#processing-datasets)
1. [Training](#training)
1. [Testing](#testing)
1. [Real-Time Viewer](#real-time-viewer)
1. [Creating Your Gaussians](#create-your-new-representations-and-rendering-pipeline)
1. [License Infomration](#license-information)
1. [Acknowledgement](#acknowledgement)
1. [Citations](#citations)


## Installation
Clone the source code of this repo.
```
git clone https://github.com/oppo-us-research/SpacetimeGaussians.git
cd SpacetimeGaussians
```

Then run the following command to install the environments with conda.
Note we will create two environments, one for preprocessing with colmap (```colmapenv```) and one for training and testing (```feature_splatting```). Training, testing and preprocessing have been tested on Ubuntu 20.04. </br>
```
bash script/setup.sh
```
Note that you may need to manually install the following packages if you encounter errors during the installation of the above command. </br>

```
conda activate feature_splatting
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch3
pip install thirdparty/gaussian_splatting/submodules/forward_lite
```

## Processing Datasets
### Neural 3D Dataset
Download the dataset from [here](https://github.com/facebookresearch/Neural_3D_Video.git).
After downloading the dataset, you can run the following command to preprocess the dataset. </br>
```
conda activate colmapenv
python script/pre_n3d.py --videopath <location>/<scene>
```
```<location>``` is the path to the dataset root folder, and ```<scene>``` is the name of a scene in the dataset. </br>

- For example if you put the dataset at ```/home/Neural3D```, and want to preprocess the ```cook_spinach``` scene, you can run the following command
```
conda activate colmapenv
python script/pre_n3d.py --videopath /home/Neural3D/cook_spinach/
```

Our codebase expects the following directory structure for the Neural 3D Dataset after preprocessing:
```

<location>
|---cook_spinach
|   |---colmap_<0>
|   |---colmap_<...>
|   |---colmap_<299>
|---flame_salmon1

```
### Technicolor Dataset
Please reach out to the authors of the paper "Dataset and Pipeline for Multi-View Light-Field Video" for access to the Technicolor dataset. </br>
Our codebase expects the following directory structure for this dataset before preprocessing:
```

<location>
|---Fabien
|   |---Fabien_undist_<00257>_<08>.png
|   |---Fabien_undist_<.....>_<..>.png
|---Birthday

```
Then run the following command to preprocess the dataset. </br>
```
conda activate colmapenv
python script/pre_technicolor.py --videopath <location>/<scene>
```
### Google Immersive Dataset (coming soon)

## Training
You can train our model by running the following command: </br>

```
conda activate feature_splatting
python train.py --quiet --eval --config config/<dataset>_<lite|full>/<scene>.json --model_path <path to save model> --source_path <location>/<scene>/colmap_0
```
In the argument to ```--config```, ```<dataset>``` can be ```n3d``` (for Neural 3D Dataset) or ```techni``` (for Technicolor Dataset), and you can choose between ```full``` model or ```lite``` model. </br>
You need 24GB GPU memory to train on the Neural 3D Dataset. </br>
You need 48GB GPU memory to train on the Technicolor Dataset. </br>
The large memory requirement is because training images are loaded into GPU memory. </br>
- For example, if you want to train the **lite** model on the first 50 frames of the ```cook_spinach``` scene in the Neural 3D Dataset, you can run the following command </br>
```
python train.py --quiet --eval --config configs/n3d_lite/cook_spinach.json --model_path log/cook_spinach_lite --source_path <location>/cook_spinach/colmap_0 
```

- If you want to train the **full** model, you can run the following command </br>

```
python train.py --quiet --eval --config configs/n3d_full/cook_spinach.json --model_path log/cook_spinach/colmap_0 --source_path <location>/cook_spinach/colmap_0 
```
Please refer to the .json config files for more options.


## Testing

- Test model on Neural 3D Dataset

```
python test.py --quiet --eval --skip_train --valloader colmapvalid --configpath config/n3d_<lite|full>/<scene>.json --model_path <path to model> --source_path <location>/<scene>/colmap_0
```

- Test model on Technicolor Dataset
```
python test.py --quiet --eval --skip_train --valloader technicolorvalid --configpath config/techni_<lite|full>/<scene>.json --model_path <path to model> --source_path <location>/<scenename>/colmap_0
```
- Test model on Google Immersive Dataset (coming soon)



## Real-Time Viewer 
The viewer is based on [SIBR](https://sibr.gitlabpages.inria.fr/) and [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). 
### Pre-built Windows Binary
Download the viewer binary from [this link](https://huggingface.co/stack93/spacetimegaussians/tree/main) and unzip it. The binary works for Windows with CUDA >= 11.0.
We also provide pre-trained models in the link. For example, [n3d_sear_steak_lite_allcam.zip](https://huggingface.co/stack93/spacetimegaussians/blob/main/n3d_sear_steak_lite_allcam.zip) contains the lite model that uses all views during training for the sear_steak scene in the Neural 3D Dataset.
### Installation from Source (coming soon)
### Running the Real-Time Viewer
After downloading the pre-built binary or installing from source, you can use the following command to run the real-time viewer. Adjust ```--iteration``` to match the training iterations of model. </br>
```
./<SIBR install dir>/bin/SIBR_gaussianViewer_app_rwdi.exe --iteration 25000 -m <path to trained model> 
``` 
The above command has beed tested on Nvidia RTX 3050 Laptop GPU + Windows 10.
- For 8K rendering, you can use the following command. </br>
```
./<SIBR install dir>/bin/SIBR_gaussianViewer_app_rwdi.exe --iteration 25000 --rendering-size 8000 4000 --force-aspect-ratio 1 -m <path to trained model> 
``` 
8K rendering has been tested on Nvidia RTX 4090 + Windows 11. 


## Create Your New Representations and Rendering Pipeline
If you want to customize our codebase for your own models, you can refer to the following steps </br>
- Step 1: Create a new Gaussian representation in this [folder](./thirdparty/gaussian_splatting/scene/). You can use ```oursfull.py``` or ```ourslite.py``` as a template. </br>
- Step 2: Create a new rendering pipeline in this [file](./thirdparty/gaussian_splatting/renderer/__init__.py). You can use the ```train_ours_full``` function as a template. </br>
- Step 3 (For new dataset, optional): Create a new dataloader in this [file](./thirdparty/gaussian_splatting/scene/__init__.py) and this [file](./thirdparty/gaussian_splatting/scene/dataset_readers.py). </br>
- Step 4: Update the intermidiate API in ```getmodel``` (for Step 1) and ```getrenderpip``` (for Step 2) functions in ```helper_train.py```.</br>


## License Information
The code in this repository (except the thirdparty folder) is licensed under MIT licence, see [LICENSE](LICENSE). thirdparty/gaussian_splatting is licensed under Gaussian-Splatting license, see [thirdparty/gaussian_splatting/LICENSE.md](thirdparty/gaussian_splatting/LICENSE.md). thirdparty/colmap is licensed under new BSD license, see [thirdparty/colmap/LICENSE.txt](thirdparty/colmap/LICENSE.txt).


## Acknowledgement
We sincerely thank the owners of the following source code repos, which are used by our released codes:
[Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[Colmap](https://github.com/colmap/colmap).
Some parts of our code referenced the following repos:
[Gaussian Splatting with Depth](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth),
[SIBR](https://gitlab.inria.fr/sibr/sibr_core.git), 
[fisheye-distortion](https://github.com/Synthesis-AI-Dev/fisheye-distortion).



## Citations
Please cite our paper if you find it useful for your research.
```
@article{li2023spacetimegaussians,
  title={Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis},
  author={Li, Zhan and Chen, Zhang and Li, Zhong and Xu, Yi},
  journal={arXiv preprint arXiv:2312.16812},
  year={2023}
}
```

Please also cite the following paper if you use Gaussian Splatting.
```
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
