Note this will install wsl on C disk. make sure your disk have enough space (> 50GB for training a model).  
I tested on Windows11 with latest nvidia driver. you don't need to install driver inside wsl.

## Step1 install WSL2 on Windows
1. if you don't have wsl2 on windows, please install wsl2 on windows in command line 
```
wsl --install
```
You should install wsl2, you can check your wsl version as following:
```
C:\Users\yourname>wsl -v
WSL version: 2.1.5.0
Kernel version: 5.15.146.1-2
WSLg version: 1.0.60
MSRDC version: 1.2.5105
Direct3D version: 1.611.1-81528511
DXCore version: 10.0.25131.1002-220531-1700.rs-onecore-base2-hyp
Windows version: 10.0.22631.2861
```

2. make a workspace dirtory in ```C:\Users\yourusername``` as following. For more commands of wsl, please see microsoft's [website](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)


```
mkdir wlsworkspace 
cd wlsworkspace
wsl 
```
after that you should be at this path inside wsl2. (```/mnt/c/``` in wsl2 is at the path of ```C:``` in windows ) 

```
yourlinuxname@yourdevice:/mnt/c/Users/yourusername/wslworkspace$
```



## Step2 install cuda toolkit inside WSL2 
1. install cuda toolkit inside wsl2, the instructions from Nvidia are [here](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local)
 We select the option for you already, just follow the selected instructions.

2. add cuda toolkit at ```PATH``` by vim editor to the end of file ```~/.bashrc``` as following. You can search how to use vim first.</p> 

```
vim ~/.bashrc
```
then type in ```i``` to insert, move the cursor to the end, paste command ```export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}``` and ```export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11.8/lib64``` tp the end two lines of file, press ```Esc```, then press ```Shift``` and ```:``` at them same time, type in ```wq``` to write the modification (w) and exit (q) the vim. 


To make new path effect, please type following in the terminal of wsl:
```
source ~/.bashrc
```

3. check installation of cudatoolkit in wsl terminal by
```
nvcc --version
```
following should be returned:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```


## Step3 install miniconda inside WSL2 
1. download minconda in workspace:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. intall it
```
bash Miniconda3-latest-Linux-x86_64.sh
```
add following path to the end of  ```~/.bashrc```. 
```
export PATH="~/miniconda3/bin:$PATH"
```

To make new path effect and enter conda, please type:
```
source ~/.bashrc
conda init
conda activate 
```

3. set conda environment to your work space (suggested, just put conda envs to the workspace incase you want to delete them.)
```
mkdir /mnt/c/Users/yourusername/wslworkspace/envs
conda config --add envs_dirs /mnt/c/Users/yourusername/wslworkspace/envs
```


## Step 4 clone our repo and follow the commands for linux in readme to setup the environments.
note that building mmcv will take sometime, just leave it there.

```
bash -i script/setup.sh
```

## Step 5 during training, if you meet error (highly possible) of ```Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory Please make sure libcudnn_cnn_infer.so.8 is in your library path!```

follow this [comment](https://github.com/pytorch/pytorch/issues/85773#issuecomment-1288033297)
## suggestions for training on windows without a large gpu memory to avoid CUDA memory error 
please set the ```r``` to 4(use smaller image size, thanks pablodawson ) in config file or set the ```duration``` to smaller values (train with fewer frames).