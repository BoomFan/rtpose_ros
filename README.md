中文版说明请参考： pose_ws/README_zh.md 

# Realtime Multi-Person Pose Estimation in ROS
This repository is used for ROS node that implement the Realtime Multi-Person Pose 2D Pose Estimation.

The original algorithm is open sourced by Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. 

At https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

## Citation

Please cite the original paper in your publications if it helps your research:

@article{cao2016realtime,

  title={Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  
  author={Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
  
  journal={arXiv preprint arXiv:1611.08050},
  
  year={2016}
  
  }

@inproceedings{wei2016cpm,

  author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
  
  booktitle = {CVPR},
  
  title = {Convolutional pose machines},
  
  year = {2016}
  
  }


# Now Let's get started!
## Compiling process
First, we need to build our own "catkin workspace". Mine is `~/pose_ws`. Please build the path and init your workspace. Then, run my commands in your terminal:
```
cd ~/pose_ws/src
git clone --recursive https://github.com/BoomFan/rtpose_ros.git
```
Since we use the C++ codes and the libraries of the original writer, we need to copy their original code and then compile. After that, compile the whole catkin workspace using my codes.

------------------------------- Attention -------------------------------

Before we compile their original code，please install the related environments, such as, cuda,cudnn,...
The original authors privided a simplified Caffe， thus we may not need to install Caffe. Later, we'll their Caffe.
Therefore, please finish all above before you goes down. Their instructions are here:https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

Or, more specifically, Step 1&2 here https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose

------------------------------- Steps for compiling their codes and then my codes -------------------------------

Step 1, Enter this directory `~/pose_ws/src/rtpose_ros`，and `git clone --recursive https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose`


Step 2, Modify this file `Makefile.config.Ubuntu14.example`  based on your needs. （Since I'm using Ubuntu14.04.) If you are using Ubuntu16, please modify `Makefile.config.Ubuntu16.example`. However, I havn't try it yet. Nothing is guaranteed.

Here is something I did：

The basic thoery is similar as a usual caffe settings.

It's better to have cudnn, otherwise I'll running out of GPU, so don't change this line:
`
USE_CUDNN := 1
`

Then, due to my cpu structure, I changed：

```
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \        
        -gencode arch=compute_50,code=sm_50 \        
        -gencode arch=compute_50,code=compute_50 \
        -gencode arch=compute_52,code=sm_52 \        
        -gencode arch=compute_60,code=sm_60 \       
        -gencode arch=compute_61,code=sm_61

# Deprecated
#CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
#        -gencode arch=compute_20,code=sm_21 \
#        -gencode arch=compute_30,code=sm_30 \
#        -gencode arch=compute_35,code=sm_35 \
#        -gencode arch=compute_50,code=sm_50 \
#        -gencode arch=compute_50,code=compute_50
```

into

```
#CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
#        -gencode arch=compute_35,code=sm_35 \
#     -gencode arch=compute_50,code=sm_50 \
#     -gencode arch=compute_50,code=compute_50 \
#     -gencode arch=compute_52,code=sm_52 \
#     -gencode arch=compute_60,code=sm_60 \
#     -gencode arch=compute_61,code=sm_61
# Deprecated
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
        -gencode arch=compute_20,code=sm_21 \
       -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_50,code=compute_50
```


Step 3, According to original authurs, theoretically, we may compile their codes by using:
```
（chmod u+x install_caffe_and_cpm.sh
./install_caffe_and_cpm.sh）
```
However, annoying errors poped out, so let's solve these problems first.

In this directory `~/pose_ws/src/rtpose_ros/caffe_rtpose/cmake/External`, this file `gflags.cmake`. Find this line：
```
set(GFLAGS_LIBRARIES ${gflags_INSTALL}/lib/libgflags.a ${CMAKE_THREAD_LIBS_INIT})
```
Add the following line after the above line:
```
set(GFLAGS_LIBRARIES ${gflags_INSTALL}/lib/libgflags.so ${CMAKE_THREAD_LIBS_INIT})
```
If you don't do so, you'll see this error:
```
/usr/bin/ld: CMakeFiles/rtpose_node.dir/src/rtpose_node.cpp.o: undefined reference to symbol '_ZN6google21ParseCommandLineFlagsEPiPPPcb'
/usr/lib/x86_64-linux-gnu/libgflags.so: error adding symbols: DSO missing from command line
```


Step 4, Now, take your time, there's one more bug here:
```
In file included from /home/roahm/pose_ws/src/rtpose_ros/caffe_rtpose/include/caffe/cpm/layers/imresize_layer.hpp:4:0,
                 from /home/roahm/pose_ws/src/rtpose_ros/src/rtpose_node.cpp:33:
/home/roahm/pose_ws/src/rtpose_ros/caffe_rtpose/include/caffe/blob.hpp:9:34: fatal error: caffe/proto/caffe.pb.h: No such file or directory
```
There is a way to deal with this error, here tells you why：http://blog.csdn.net/xmzwlw/article/details/48270225

Basically, is to creat this directory`~/pose_ws/src/rtpose_ros/caffe_rtpose/include/caffe/proto`, and run the following comands one by one：
```
cd ~/pose_ws/src/rtpose_ros/caffe_rtpose/src/caffe/proto
protoc --cpp_out=/home/roahm/pose_ws/src/rtpose_ros/caffe_rtpose/include/caffe/proto caffe.proto
```
(Please change the path into your catkin workspace if you need to do so.)


Step 5, Now, go back to `~/pose_ws/src/rtpose_ros/caffe_rtpose`, we can finally compile their original codes:
```
chmod u+x install_caffe_and_cpm.sh
./install_caffe_and_cpm.sh
```



Step 6, If you already `git clone` my codes, you already have my `CMakeLists.txt` and `package.xml` under directory `~/pose_ws/src/rtpose_ros`


Step 7, Now, edit the cpp node file that you want to run. For example, in this file `~/pose_ws/src/rtpose_ros/src/rtpose_node.cpp`, you may need to change the topic name that you want to subscribe and the topic name that you want to publish into. As well as other parameters like, num of GPU, caffemodel path, camera resolution(your ROS topic resolution), etc.

Then `cd ~/pose_ws` and `catkin_make`


Step 8, Congratulations! You may run the ROS node now!
```
rosrun rtpose_ros rtpose_node
```
