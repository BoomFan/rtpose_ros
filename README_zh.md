English tutorial of my codes is here : pose_ws/README.md 

# Realtime Multi-Person Pose Estimation in ROS 
# (在ROS中实现多人姿态识别)
本代码库用于在ROS中实现实时多人姿态识别算法（Realtime Multi-Person Pose 2D Pose Estimation）。

源代码由Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. 提供。他们的官方代码库如下

https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

## 引用
算法本身不是我的原创，如本算法对您有帮助，请引用原作者的文章:

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


# 下面进入正题
## 运行步骤
首先我们需要建立自己的catkin workspace, 本案例中假定是~/pose_ws路径，请先自行创建路径并初始化。然后
```
cd ~/pose_ws/src
git clone --recursive https://github.com/BoomFan/ROSposeEstimation.git
```
由于本代码调用了原作者提供的C++代码及其library.所以我们需要先拷贝原作者的代码，编译成功后，再运用我的代码，对整个catkin workspace进行编译。

------------------------------- 特别注意 -------------------------------

在编译原作者代码之前，请先安装好相关环境。如：cuda,cudnn等。由于他们提供了删减版的Caffe，所以可以暂时不用安装Caffe.我们之后会编译原作者所提供的Caffe.所以请务必先完成这一准备过程，具体细节请参考原地址https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

------------------------------- 本代码库具体使用步骤如下 -------------------------------

步骤1，先进入路径~/pose_ws/src/rtpose_ros，把https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose 里面的东西git clone 下来，

步骤2，然后把Makefile.config.Ubuntu14.example文件配置好（因这里我是用的是Ubuntu14.04,如果您使用Ubuntu16,请修改Makefile.config.Ubuntu16.example，但我并未测试过Ubuntu16），这里将我自己的改动记录如下：

基本思路与caffe配置差不多

最好还是使用cudnn,不然我的电脑会提示gpu不够，所以这一行不变
`
USE_CUDNN := 1
`

由于cpu结构的原因，我将：

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

改为

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

步骤3，然后注意了！！！！！理论上，根据作者的意思，已经可以开始用这两个命令
```
（chmod u+x install_caffe_and_cpm.sh
./install_caffe_and_cpm.sh）
```
进行编译了。但由于我遇到了许多编译错误，在此先解决一个报错。

先改这个路径下~/pose_ws/src/rtpose_ros/caffe_rtpose/cmake/External的gflags.cmake文件中的这一行：
```
set(GFLAGS_LIBRARIES ${gflags_INSTALL}/lib/libgflags.a ${CMAKE_THREAD_LIBS_INIT})
```
后面加入这一行
```
set(GFLAGS_LIBRARIES ${gflags_INSTALL}/lib/libgflags.so ${CMAKE_THREAD_LIBS_INIT})
```
因为如果不加这一行，后面会有很恶心很恶心的报错，我花了两天时间才搞清楚怎么回事。就是下面这两行报错：
```
/usr/bin/ld: CMakeFiles/rtpose_node.dir/src/rtpose_node.cpp.o: undefined reference to symbol '_ZN6google21ParseCommandLineFlagsEPiPPPcb'
/usr/lib/x86_64-linux-gnu/libgflags.so: error adding symbols: DSO missing from command line
```

步骤4，稍安勿躁，我们再来处理一个报错：
```
In file included from /home/roahm/pose_ws/src/rtpose_ros/caffe_rtpose/include/caffe/cpm/layers/imresize_layer.hpp:4:0,
                 from /home/roahm/pose_ws/src/rtpose_ros/src/rtpose_node.cpp:33:
/home/roahm/pose_ws/src/rtpose_ros/caffe_rtpose/include/caffe/blob.hpp:9:34: fatal error: caffe/proto/caffe.pb.h: No such file or directory
```
解决方法看这里：http://blog.csdn.net/xmzwlw/article/details/48270225

核心思想就是要在`~/pose_ws/src/rtpose_ros/caffe_rtpose/include/caffe`里面建一个叫做`proto`的文件夹，然后要在这个文件夹里生成`caffe.pb.h`和`caffe.pb.cc`这样的两个文件。简言之：
```
cd ~/pose_ws/src/rtpose_ros/caffe_rtpose/src/caffe/proto
protoc --cpp_out=/home/roahm/pose_ws/src/rtpose_ros/caffe_rtpose/include/caffe/proto caffe.proto
```
请根据自己所需修改catkin workspace路径.


步骤5，在路径`~/pose_ws/src/rtpose_ros/caffe_rtpose`下:
```
chmod u+x install_caffe_and_cpm.sh
./install_caffe_and_cpm.sh
```
就能把原代码库编译好了

步骤6，如果你克隆了我的代码，我已配置好`~/pose_ws/src/rtpose_ros`路径下所需的`CMakeLists.txt`以及`package.xml`文件

步骤7, 然后修改你要运行的node的cpp文件，如：`~/pose_ws/src/rtpose_ros/src/rtpose_node.cpp`文件。修改里面要发布的topic名称和要订阅的topic名称，以及其他参数，例如GPU数量，caffemodel存储的位置，等。

然后再到`~/pose_ws`路径下
```
catkin_make
```

步骤8,恭喜你编译成功，运行下列代码以启动ROS节点
```
rosrun rtpose rtpose_node
```

