#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <utility> //std::pair

#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <stdio.h>  // snprintf
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>

#include <boost/thread/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <gflags/gflags.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/cpm/frame.h"
#include "caffe/cpm/layers/imresize_layer.hpp"
#include "caffe/cpm/layers/nms_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/blocking_queue.hpp"
// #include "caffe/util/render_functions.hpp"
// #include "caffe/blob.hpp"
// #include "caffe/common.hpp"
// #include "caffe/proto/caffe.pb.h"
// #include "caffe/util/db.hpp"
// #include "caffe/util/io.hpp"
// #include "caffe/util/benchmark.hpp"

#include "rtpose/modelDescriptor.h"
#include "rtpose/modelDescriptorFactory.h"
#include "rtpose/renderFunctions.h"

//ROS Includes
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/image_encodings.h>
#include <highgui.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <std_msgs/String.h>
#include <sstream>
// ROS time_synchronizer for subscribing wultiple topics at the same time
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

// Add some CUDA stuff  --Fan Bu
#include <cuda.h>
#include <cuda_runtime.h>

// Add libs to publish result into MultiArray
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Int32MultiArray.h"
#include "std_msgs/Float32MultiArray.h"

// My own message type
#include "rtpose_ros/Detection.h"
#include "Hungarian.h"

// PCL includes
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

// Flags (rtpose.bin --help)
DEFINE_bool(fullscreen,             false,          "Run in fullscreen mode (press f during runtime to toggle)");
DEFINE_int32(part_to_show,          0,              "Part to show from the start.");
DEFINE_string(write_frames,         "",             "Write frames with format prefix%06d.jpg");
DEFINE_bool(no_frame_drops,         false,          "Dont drop frames.");
DEFINE_string(write_json,           "",             "Write joint data with json format as prefix%06d.json");
DEFINE_int32(camera,                0,              "The camera index for VideoCapture.");
DEFINE_string(video,                "",             "Use a video file instead of the camera.");
DEFINE_string(image_dir,            "",             "Process a directory of images.");
DEFINE_int32(start_frame,           0,              "Skip to frame # of video");

DEFINE_string(caffemodel, "/home/roahm/pose_ws/src/rtpose_ros/caffe_rtpose/model/coco/pose_iter_440000.caffemodel", "Caffe model.");
DEFINE_string(caffeproto, "/home/roahm/pose_ws/src/rtpose_ros/caffe_rtpose/model/coco/pose_deploy_linevec.prototxt", "Caffe deploy prototxt.");
//DEFINE_string(caffemodel, "model/coco/pose_iter_440000.caffemodel", "Caffe model.");
//DEFINE_string(caffeproto, "model/coco/pose_deploy_linevec.prototxt", "Caffe deploy prototxt.");
// DEFINE_string(caffemodel, "/home/roahm/pose_ws/src/rtpose/caffe_rtpose/model/mpi/pose_iter_160000.caffemodel", "Caffe model.");
// DEFINE_string(caffeproto, "/home/roahm/pose_ws/src/rtpose/caffe_rtpose/model/mpi/pose_deploy_linevec.prototxt", "Caffe deploy prototxt.");
// DEFINE_string(caffemodel, "model/mpi/pose_iter_160000.caffemodel", "Caffe model.");
// DEFINE_string(caffeproto, "model/mpi/pose_deploy_linevec.prototxt", "Caffe deploy prototxt.");
DEFINE_string(resolution,           "1280x720",     "The image resolution (display).");
DEFINE_string(net_resolution,       "656x368",      "Multiples of 16.");
// DEFINE_string(camera_resolution,    "1920x1080",     "Size of the camera frames to ask for.");// For ubs_cam
DEFINE_string(camera_resolution,    "1280x720",     "Size of the camera frames to ask for.");// For ZED camera
DEFINE_int32(start_device,          0,              "GPU device start number.");
DEFINE_int32(num_gpu,               2,              "The number of GPU devices to use.");
DEFINE_double(start_scale,          1,              "Initial scale. Must cv::Match net_resolution");
DEFINE_double(scale_gap,            0.3,            "Scale gap between scales. No effect unless num_scales>1");
DEFINE_int32(num_scales,            1,              "Number of scales to average");
DEFINE_bool(no_display,             false,          "Do not open a display window.");

// Global parameters
int DISPLAY_RESOLUTION_WIDTH;
int DISPLAY_RESOLUTION_HEIGHT;
int CAMERA_FRAME_WIDTH;
int CAMERA_FRAME_HEIGHT;
int NET_RESOLUTION_WIDTH;
int NET_RESOLUTION_HEIGHT;
int BATCH_SIZE;
double SCALE_GAP;
double START_SCALE;
int NUM_GPU;
std::string PERSON_DETECTOR_CAFFEMODEL; //person detector
std::string PERSON_DETECTOR_PROTO;      //person detector
std::string POSE_ESTIMATOR_PROTO;       //pose estimator
const auto MAX_PEOPLE = RENDER_MAX_PEOPLE;  // defined in render_functions.hpp
const auto BOX_SIZE = 368;
const auto BUFFER_SIZE = 4;    //affects latency
const auto MAX_NUM_PARTS = 70;

// Start ---------- My own define ----------Fan Bu
//const std::string RECEIVE_LEFT_IMG_TOPIC_NAME = "camera/rgb/image_raw";
const std::string RECEIVE_LEFT_IMG_TOPIC_NAME = "camera/left/image_rect_color";
const std::string RECEIVE_RIGHT_IMG_TOPIC_NAME = "camera/right/image_rect_color";
// const std::string RECEIVE_LEFT_IMG_TOPIC_NAME = "/usb_cam/image_raw";
const std::string PUBLISH_LEFT_IMG_TOPIC_NAME = "pose_estimate/image_left";
const std::string PUBLISH_RIGHT_IMG_TOPIC_NAME = "pose_estimate/image_right";
// const std::string PUBLISH_STR_TOPIC_NAME = "pose_estimate/str";
// const std::string PUBLISH_ARY_TOPIC_NAME = "pose_estimate/ary";
const std::string PUBLISH_DETECTION_NAME = "pose_estimate/detection";
const std::string PUBLISH_3D_PCD_NAME = "pose_estimate/pcd";
std::string camera_link_name = "camera_link";

cv::Mat final_img;
image_transport::Publisher poseLeftImagePublisher;
image_transport::Publisher poseRightImagePublisher;
std_msgs::Header header;
// ros::Publisher poseStrPublisher;// Buffer size 1000
// ros::Publisher poseAryPublisher;// To publish estimation array
ros::Publisher poseDetectionPublisher;// To publish Detection of stareo images(customized message format)
ros::Publisher cloudRGBPublisher;


int display_counter = 1;
double last_pop_time;

// Flags for stereo processing
bool left_processing = true;
bool right_processing = true;
// End ------------- My own define ---------Fan Bu

// global queues for I/O
struct Global {

    caffe::BlockingQueue<Frame> input_queue; //have to pop
    caffe::BlockingQueue<Frame> output_queue; //have to pop
    caffe::BlockingQueue<Frame> output_queue_ordered;
    caffe::BlockingQueue<Frame> output_queue_mated;
    caffe::BlockingQueue<Frame> input_right_queue;              // Added by Fan Bu
    caffe::BlockingQueue<Frame> output_right_queue;             // Added by Fan Bu
    caffe::BlockingQueue<Frame> output_right_queue_ordered;     // Added by Fan Bu
    caffe::BlockingQueue<Frame> output_right_queue_mated;       // Added by Fan Bu
    std::priority_queue<int, std::vector<int>, std::greater<int> > dropped_index;
    std::vector< std::string > image_list;
    std::mutex mutex;
    int part_to_show;
    bool quit_threads;
    // Parameters
    float nms_threshold;
    int connect_min_subset_cnt;
    float connect_min_subset_score;
    float connect_inter_threshold;
    int connect_inter_min_above_threshold;

    struct UIState {
        UIState() :
            is_fullscreen(0),
            is_video_paused(0),
            is_shift_down(0),
            is_googly_eyes(0),
            current_frame(0),
            seek_to_frame(-1),
            fps(0) {}
        bool is_fullscreen;
        bool is_video_paused;
        bool is_shift_down;
        bool is_googly_eyes;
        int current_frame;
        int seek_to_frame;
        double fps;
    };
    UIState uistate;
 };

// network copy for each gpu thread
struct NetCopy {
    caffe::Net<float> *person_net;
    std::vector<int> num_people;
    int nblob_person;
    int nms_max_peaks;
    int nms_num_parts;
    std::unique_ptr<ModelDescriptor> up_model_descriptor;
    float* canvas; // GPU memory
    float* joints; // GPU memory
};

struct ColumnCompare
{
    bool operator()(const std::vector<double>& lhs,
                    const std::vector<double>& rhs) const
    {
        return lhs[2] > rhs[2];
        //return lhs[0] > rhs[0];
    }
};

Global global;
std::vector<NetCopy> net_copies;

int rtcpm_stereo();
bool handleKey(int c);
void warmup(int);
void process_and_pad_image(float* target, cv::Mat oriImg, int tw, int th, bool normalize);


double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time,NULL)) {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
    //return (double)time.tv_usec;
}

void warmup(int device_id) {
    int logtostderr = FLAGS_logtostderr;

    LOG(INFO) << "Setting GPU " << device_id;

    caffe::Caffe::SetDevice(device_id); //cudaSetDevice(device_id) inside
    caffe::Caffe::set_mode(caffe::Caffe::GPU); //

    LOG(INFO) << "GPU " << device_id << ": copying to person net";
    FLAGS_logtostderr = 0;
    net_copies[device_id].person_net = new caffe::Net<float>(PERSON_DETECTOR_PROTO, caffe::TEST);
    net_copies[device_id].person_net->CopyTrainedLayersFrom(PERSON_DETECTOR_CAFFEMODEL);

    net_copies[device_id].nblob_person = net_copies[device_id].person_net->blob_names().size();
    net_copies[device_id].num_people.resize(BATCH_SIZE);
    const std::vector<int> shape { {BATCH_SIZE, 3, NET_RESOLUTION_HEIGHT, NET_RESOLUTION_WIDTH} };

    net_copies[device_id].person_net->blobs()[0]->Reshape(shape);
    net_copies[device_id].person_net->Reshape();
    FLAGS_logtostderr = logtostderr;

    caffe::NmsLayer<float> *nms_layer = (caffe::NmsLayer<float>*)net_copies[device_id].person_net->layer_by_name("nms").get();
    net_copies[device_id].nms_max_peaks = nms_layer->GetMaxPeaks();


    caffe::ImResizeLayer<float> *resize_layer =
        (caffe::ImResizeLayer<float>*)net_copies[device_id].person_net->layer_by_name("resize").get();

    resize_layer->SetStartScale(START_SCALE);
    resize_layer->SetScaleGap(SCALE_GAP);
    LOG(INFO) << "start_scale = " << START_SCALE;

    net_copies[device_id].nms_max_peaks = nms_layer->GetMaxPeaks();

    net_copies[device_id].nms_num_parts = nms_layer->GetNumParts();
    CHECK_LE(net_copies[device_id].nms_num_parts, MAX_NUM_PARTS)
        << "num_parts in NMS layer (" << net_copies[device_id].nms_num_parts << ") "
        << "too big ( MAX_NUM_PARTS )";
// printf("check point 1\n");
    if (net_copies[device_id].nms_num_parts==15) {
        // printf("check point 1.1\n");
        ModelDescriptorFactory::createModelDescriptor(ModelDescriptorFactory::Type::MPI_15, net_copies[device_id].up_model_descriptor);
        global.nms_threshold = 0.2;
        global.connect_min_subset_cnt = 3;
        global.connect_min_subset_score = 0.4;
        global.connect_inter_threshold = 0.01;
        global.connect_inter_min_above_threshold = 8;
        LOG(INFO) << "Selecting MPI model.";
    } else if (net_copies[device_id].nms_num_parts==18) {
        // printf("check point 1.2\n");
        printf("device_id is %i \n", device_id);
        ModelDescriptorFactory::createModelDescriptor(ModelDescriptorFactory::Type::COCO_18, net_copies[device_id].up_model_descriptor);
        // printf("check point 1.2.1\n");
        global.nms_threshold = 0.05;
        global.connect_min_subset_cnt = 3;
        global.connect_min_subset_score = 0.4;
        global.connect_inter_threshold = 0.050;
        global.connect_inter_min_above_threshold = 9;
        // printf("check point 1.2.2\n");
    } else {
        // printf("check point 1.3\n");
        CHECK(0) << "Unknown number of parts! Couldn't set model";
    }
// printf("check point 2\n");
    //dry run
    LOG(INFO) << "Dry running...";
    net_copies[device_id].person_net->ForwardFrom(0);
    LOG(INFO) << "Success.";
    // printf("check point 3\n");
    cudaMalloc(&net_copies[device_id].canvas, DISPLAY_RESOLUTION_WIDTH * DISPLAY_RESOLUTION_HEIGHT * 3 * sizeof(float));
    // printf("check point 4\n");
    cudaMalloc(&net_copies[device_id].joints, MAX_NUM_PARTS*3*MAX_PEOPLE * sizeof(float) );
    // printf("check point 5\n");
}

void process_and_pad_image(float* target, cv::Mat oriImg, int tw, int th, bool normalize) {
    int ow = oriImg.cols;
    int oh = oriImg.rows;
    int offset2_target = tw * th;

    int padw = (tw-ow)/2;
    int padh = (th-oh)/2;
    //LOG(ERROR) << " padw " << padw << " padh " << padh;
    CHECK_GE(padw,0) << "Image too big for target size.";
    CHECK_GE(padh,0) << "Image too big for target size.";
    //parallel here
    unsigned char* pointer = (unsigned char*)(oriImg.data);

    for(int c = 0; c < 3; c++) {
        for(int y = 0; y < th; y++) {
            int oy = y - padh;
            for(int x = 0; x < tw; x++) {
                int ox = x - padw;
                if (ox>=0 && ox < ow && oy>=0 && oy < oh ) {
                    if (normalize)
                        target[c * offset2_target + y * tw + x] = float(pointer[(oy * ow + ox) * 3 + c])/256.0f - 0.5f;
                    else
                        target[c * offset2_target + y * tw + x] = float(pointer[(oy * ow + ox) * 3 + c]);
                }
                else {
                    target[c * offset2_target + y * tw + x] = 0;
                }
            }
        }
    }
}

void render(int gid, float *heatmaps /*GPU*/) {
    float* centers = 0;
    float* poses    = net_copies[gid].joints;

    double tic = get_wall_time();
    if (net_copies[gid].up_model_descriptor->get_number_parts()==15) {
        render_mpi_parts(net_copies[gid].canvas, DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT, NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT,
        heatmaps, BOX_SIZE, centers, poses, net_copies[gid].num_people, global.part_to_show);
    } else if (net_copies[gid].up_model_descriptor->get_number_parts()==18) {
        if (global.part_to_show-1<=net_copies[gid].up_model_descriptor->get_number_parts()) {
            render_coco_parts(net_copies[gid].canvas,
            DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT,
            NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT,
            heatmaps, BOX_SIZE, centers, poses,
            net_copies[gid].num_people, global.part_to_show, global.uistate.is_googly_eyes);
        } else {
            int aff_part = ((global.part_to_show-1)-net_copies[gid].up_model_descriptor->get_number_parts()-1)*2;
            int num_parts_accum = 1;
            if (aff_part==0) {
                num_parts_accum = 19;
            } else {
                aff_part = aff_part-2;
                }
                aff_part += 1+net_copies[gid].up_model_descriptor->get_number_parts();
                render_coco_aff(net_copies[gid].canvas, DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT, NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT,
                heatmaps, BOX_SIZE, centers, poses, net_copies[gid].num_people, aff_part, num_parts_accum);
        }
    }
    VLOG(2) << "Render time " << (get_wall_time()-tic)*1000.0 << " ms.";
}

void* getFrameFromDir(void *i) {
    int global_counter = 1;
    int frame_counter = 0;
    cv::Mat image_uchar;
    cv::Mat image_uchar_orig;
    cv::Mat image_uchar_prev;
    while(1) {
        if (global.quit_threads) break;
        // If the queue is too long, wait for a bit
        if (global.input_queue.size()>10) {
            usleep(10*1000.0);
            continue;
        }

        // Keep a count of how many frames we've seen in the video
        frame_counter++;

        // This should probably be protected.
        global.uistate.current_frame = frame_counter-1;

        std::string filename = global.image_list[global.uistate.current_frame];
        image_uchar_orig = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
        double scale = 0;
        if (image_uchar_orig.cols/(double)image_uchar_orig.rows>DISPLAY_RESOLUTION_WIDTH/(double)DISPLAY_RESOLUTION_HEIGHT) {
            scale = DISPLAY_RESOLUTION_WIDTH/(double)image_uchar_orig.cols;
        } else {
            scale = DISPLAY_RESOLUTION_HEIGHT/(double)image_uchar_orig.rows;
        }
        cv::Mat M = cv::Mat::eye(2,3,CV_64F);
        M.at<double>(0,0) = scale;
        M.at<double>(1,1) = scale;
        cv::warpAffine(image_uchar_orig, image_uchar, M,
                             cv::Size(DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT),
                             CV_INTER_CUBIC,
                             cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        // resize(image_uchar, image_uchar, cv::Size(new_width, new_height), 0, 0, CV_INTER_CUBIC);
        image_uchar_prev = image_uchar;

        if ( image_uchar.empty() ) continue;

        Frame frame;
        frame.ori_width = image_uchar_orig.cols;
        frame.ori_height = image_uchar_orig.rows;
        frame.index = global_counter++;
        frame.video_frame_number = global.uistate.current_frame;
        frame.data_for_wrap = new unsigned char [DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3]; //fill after process
        frame.data_for_mat = new float [DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3];
        process_and_pad_image(frame.data_for_mat, image_uchar, DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT, 0);

        frame.scale = scale;
        //pad and transform to float
        int offset = 3 * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
        frame.data = new float [BATCH_SIZE * offset];
        int target_width, target_height;
        cv::Mat image_temp;
        //LOG(ERROR) << "frame.index: " << frame.index;
        for(int i=0; i < BATCH_SIZE; i++) {
            float scale = START_SCALE - i*SCALE_GAP;
            target_width = 16 * ceil(NET_RESOLUTION_WIDTH * scale /16);
            target_height = 16 * ceil(NET_RESOLUTION_HEIGHT * scale /16);

            CHECK_LE(target_width, NET_RESOLUTION_WIDTH);
            CHECK_LE(target_height, NET_RESOLUTION_HEIGHT);

            resize(image_uchar, image_temp, cv::Size(target_width, target_height), 0, 0, CV_INTER_AREA);
            process_and_pad_image(frame.data + i * offset, image_temp, NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT, 1);
        }
        frame.commit_time = get_wall_time();
        frame.preprocessed_time = get_wall_time();

        global.input_queue.push(frame);

        // If we reach the end of a video, loop
        if (frame_counter >= global.image_list.size()) {
            LOG(INFO) << "Done, exiting. # frames: " << frame_counter;
            // Wait until the queues are clear before exiting
            while (global.input_queue.size()
                    || global.output_queue.size()
                    || global.output_queue_ordered.size()) {
                // Should actually wait until they finish writing to disk
                // This could exit before the last frame is written.
                usleep(1000*1000.0);
                continue;
            }
            global.quit_threads = true;
            global.uistate.is_video_paused = true;
        }
    }
    return nullptr;
}

void* getFrameFromCam(void *i) {
    cv::VideoCapture cap;
    double target_frame_time = 0;
    double target_frame_rate = 0;
    if (!FLAGS_image_dir.empty()) {
        return getFrameFromDir(i);
    }

    if (FLAGS_video.empty()) {
        CHECK(cap.open(FLAGS_camera)) << "Couldn't open camera " << FLAGS_camera;
        cap.set(CV_CAP_PROP_FRAME_WIDTH,CAMERA_FRAME_WIDTH);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT,CAMERA_FRAME_HEIGHT);
    } else {
        CHECK(cap.open(FLAGS_video)) << "Couldn't open video file " << FLAGS_video;
        target_frame_rate = cap.get(CV_CAP_PROP_FPS);
        target_frame_time = 1.0/target_frame_rate;
        if (FLAGS_start_frame) {
            cap.set(CV_CAP_PROP_POS_FRAMES, FLAGS_start_frame);
        }
    }

    int global_counter = 1;
    int frame_counter = 0;
    cv::Mat image_uchar;
    cv::Mat image_uchar_orig;
    cv::Mat image_uchar_prev;
    double last_frame_time = -1;
    while(1) {
        if (global.quit_threads) {
            break;
        }
        if (!FLAGS_video.empty() && FLAGS_no_frame_drops) {
            // If the queue is too long, wait for a bit
            if (global.input_queue.size()>10) {
                usleep(10*1000.0);
                continue;
            }
        }
        cap >> image_uchar_orig;
        // Keep a count of how many frames we've seen in the video
        if (!FLAGS_video.empty()) {
            if (global.uistate.seek_to_frame!=-1) {
                cap.set(CV_CAP_PROP_POS_FRAMES, global.uistate.current_frame);
                global.uistate.seek_to_frame = -1;
            }
            frame_counter = cap.get(CV_CAP_PROP_POS_FRAMES);

            VLOG(3) << "Frame: " << frame_counter << " / " << cap.get(CV_CAP_PROP_FRAME_COUNT);
            // This should probably be protected.
            global.uistate.current_frame = frame_counter-1;
            if (global.uistate.is_video_paused) {
                cap.set(CV_CAP_PROP_POS_FRAMES, frame_counter-1);
                frame_counter -= 1;
            }

            // Sleep to get the right frame rate.
            double cur_frame_time = get_wall_time();
            double interval = (cur_frame_time-last_frame_time);
            VLOG(3) << "cur_frame_time " << (cur_frame_time);
            VLOG(3) << "last_frame_time " << (last_frame_time);
            VLOG(3) << "cur-last_frame_time " << (cur_frame_time - last_frame_time);
            VLOG(3) << "Video target frametime " << 1.0/target_frame_time
                            << " read frametime " << 1.0/interval;
            if (interval<target_frame_time) {
                VLOG(3) << "Sleeping for " << (target_frame_time-interval)*1000.0;
                usleep((target_frame_time-interval)*1000.0*1000.0);
                cur_frame_time = get_wall_time();
            }
            last_frame_time = cur_frame_time;
        } else {
            // From camera, just increase counter.
            if (global.uistate.is_video_paused) {
                image_uchar_orig = image_uchar_prev;
            }
            image_uchar_prev = image_uchar_orig;
            frame_counter++;
        }

        // TODO: The entire scaling code should be rewritten and better matched
        // to the imresize_layer. Confusingly, for the demo, there's an intermediate
        // display resolution to which the original image is resized.
        double scale = 0;
        if (image_uchar_orig.cols/(double)image_uchar_orig.rows>DISPLAY_RESOLUTION_WIDTH/(double)DISPLAY_RESOLUTION_HEIGHT) {
            scale = DISPLAY_RESOLUTION_WIDTH/(double)image_uchar_orig.cols;
        } else {
            scale = DISPLAY_RESOLUTION_HEIGHT/(double)image_uchar_orig.rows;
        }
        VLOG(4) << "Scale to DISPLAY_RESOLUTION_WIDTH/HEIGHT: " << scale;
        cv::Mat M = cv::Mat::eye(2,3,CV_64F);
        M.at<double>(0,0) = scale;
        M.at<double>(1,1) = scale;
        warpAffine(image_uchar_orig, image_uchar, M,
                             cv::Size(DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT),
                             CV_INTER_CUBIC,
                             cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        // resize(image_uchar, image_uchar, Size(new_width, new_height), 0, 0, CV_INTER_CUBIC);
        image_uchar_prev = image_uchar_orig;

        if ( image_uchar.empty() )
            continue;

        Frame frame;
        frame.scale = scale;
        frame.index = global_counter++;
        frame.video_frame_number = global.uistate.current_frame;
        frame.data_for_wrap = new unsigned char [DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3]; //fill after process
        frame.data_for_mat = new float [DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3];
        process_and_pad_image(frame.data_for_mat, image_uchar, DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT, 0);

        //pad and transform to float
        int offset = 3 * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
        frame.data = new float [BATCH_SIZE * offset];
        int target_width;
        int target_height;
        cv::Mat image_temp;
        for(int i=0; i < BATCH_SIZE; i++) {
            float scale = START_SCALE - i*SCALE_GAP;
            target_width = 16 * ceil(NET_RESOLUTION_WIDTH * scale /16);
            target_height = 16 * ceil(NET_RESOLUTION_HEIGHT * scale /16);

            CHECK_LE(target_width, NET_RESOLUTION_WIDTH);
            CHECK_LE(target_height, NET_RESOLUTION_HEIGHT);

            cv::resize(image_uchar, image_temp, cv::Size(target_width, target_height), 0, 0, CV_INTER_AREA);
            process_and_pad_image(frame.data + i * offset, image_temp, NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT, 1);
        }
        frame.commit_time = get_wall_time();
        frame.preprocessed_time = get_wall_time();

        global.input_queue.push(frame);

        // If we reach the end of a video, loop
        if (!FLAGS_video.empty() && frame_counter >= cap.get(CV_CAP_PROP_FRAME_COUNT)) {
            if (!FLAGS_write_frames.empty()) {
                LOG(INFO) << "Done, exiting. # frames: " << frame_counter;
                // This is the last frame (also the last emmitted frame)
                // Wait until the queues are clear before exiting
                while (global.input_queue.size()
                            || global.output_queue.size()
                            || global.output_queue_ordered.size()) {
                    // Should actually wait until they finish writing to disk.
                    // This could exit before the last frame is written.
                    usleep(1000*1000.0);
                    continue;
                }
                global.quit_threads = true;
                global.uistate.is_video_paused = true;
            } else {
                LOG(INFO) << "Looping video after " << cap.get(CV_CAP_PROP_FRAME_COUNT) << " frames";
                cap.set(CV_CAP_PROP_POS_FRAMES, 0);
            }
        }
    }
    return nullptr;
}

void getFrameFromROS(const sensor_msgs::ImageConstPtr& msg) {
    cv::Mat img;
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "rgb8");
    printf("Received one new Image.\n");
    //cv::imwrite("rgb.png", cv_ptr->image);
    //cv::Mat img = cv_ptr->image;
    img = cv_ptr->image;
    double target_frame_time = 0;
    double target_frame_rate = 0;

    int global_counter = 1;
    int frame_counter = 0;
    cv::Mat image_uchar;
    cv::Mat image_uchar_orig;
    cv::Mat image_uchar_prev;
    double last_frame_time = -1;


        image_uchar_orig = img;

        image_uchar_prev = image_uchar_orig;
        frame_counter++;


        // TODO: The entire scaling code should be rewritten and better matched
        // to the imresize_layer. Confusingly, for the demo, there's an intermediate
        // display resolution to which the original image is resized.
        double scale = 0;
        if (image_uchar_orig.cols/(double)image_uchar_orig.rows>DISPLAY_RESOLUTION_WIDTH/(double)DISPLAY_RESOLUTION_HEIGHT) {
            scale = DISPLAY_RESOLUTION_WIDTH/(double)image_uchar_orig.cols;
        } else {
            scale = DISPLAY_RESOLUTION_HEIGHT/(double)image_uchar_orig.rows;
        }
        VLOG(4) << "Scale to DISPLAY_RESOLUTION_WIDTH/HEIGHT: " << scale;
        cv::Mat M = cv::Mat::eye(2,3,CV_64F);
        M.at<double>(0,0) = scale;
        M.at<double>(1,1) = scale;
        warpAffine(image_uchar_orig, image_uchar, M,
                             cv::Size(DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT),
                             CV_INTER_CUBIC,
                             cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        // resize(image_uchar, image_uchar, Size(new_width, new_height), 0, 0, CV_INTER_CUBIC);
        image_uchar_prev = image_uchar_orig;

        Frame frame;
        frame.scale = scale;
        frame.index = global_counter++;
        frame.video_frame_number = global.uistate.current_frame;
        frame.data_for_wrap = new unsigned char [DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3]; //fill after process
        frame.data_for_mat = new float [DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3];
        process_and_pad_image(frame.data_for_mat, image_uchar, DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT, 0);

        //pad and transform to float
        int offset = 3 * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
        frame.data = new float [BATCH_SIZE * offset];
        int target_width;
        int target_height;
        cv::Mat image_temp;
        for(int i=0; i < BATCH_SIZE; i++) {
            float scale = START_SCALE - i*SCALE_GAP;
            target_width = 16 * ceil(NET_RESOLUTION_WIDTH * scale /16);
            target_height = 16 * ceil(NET_RESOLUTION_HEIGHT * scale /16);

            CHECK_LE(target_width, NET_RESOLUTION_WIDTH);
            CHECK_LE(target_height, NET_RESOLUTION_HEIGHT);

            cv::resize(image_uchar, image_temp, cv::Size(target_width, target_height), 0, 0, CV_INTER_AREA);
            process_and_pad_image(frame.data + i * offset, image_temp, NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT, 1);
        }
        frame.commit_time = get_wall_time();
        frame.preprocessed_time = get_wall_time();

        global.input_queue.push(frame);

    // return nullptr;
}


void getStereoFrameFromROS(const sensor_msgs::ImageConstPtr& left_image, const sensor_msgs::ImageConstPtr& right_image) {
    // ROS_INFO("Now left_processing is: %s ; right_processing is: %s.", left_processing ? "true" : "false", right_processing ? "true" : "false");
    if (!left_processing && !right_processing){
        left_processing = true;
        right_processing = true;
        cv::Mat img_left;
        cv::Mat img_right;
        cv_bridge::CvImagePtr cv_ptr_left = cv_bridge::toCvCopy(left_image, "rgb8");
        cv_bridge::CvImagePtr cv_ptr_right = cv_bridge::toCvCopy(right_image, "rgb8");
        ROS_INFO("Received New Stereo Images.");
        //cv::imwrite("rgb.png", cv_ptr->image);
        //cv::Mat img = cv_ptr->image;
        img_left = cv_ptr_left->image;
        img_right = cv_ptr_right->image;
        double target_frame_time = 0;
        double target_frame_rate = 0;

        int global_counter = 1;
        int frame_counter = 0;
        cv::Mat image_uchar;
        cv::Mat image_uchar_orig;
        cv::Mat image_uchar_prev;
        double last_frame_time = -1;

        for (int j=0; j < 2; j++){
            if(j==0){
                image_uchar_orig = img_left;
            }
            else{
                image_uchar_orig = img_right;
            }

            image_uchar_prev = image_uchar_orig;
            frame_counter++;


            // TODO: The entire scaling code should be rewritten and better matched
            // to the imresize_layer. Confusingly, for the demo, there's an intermediate
            // display resolution to which the original image is resized.
            double scale = 0;
            if (image_uchar_orig.cols/(double)image_uchar_orig.rows>DISPLAY_RESOLUTION_WIDTH/(double)DISPLAY_RESOLUTION_HEIGHT) {
                scale = DISPLAY_RESOLUTION_WIDTH/(double)image_uchar_orig.cols;
            } else {
                scale = DISPLAY_RESOLUTION_HEIGHT/(double)image_uchar_orig.rows;
            }
            VLOG(4) << "Scale to DISPLAY_RESOLUTION_WIDTH/HEIGHT: " << scale;
            cv::Mat M = cv::Mat::eye(2,3,CV_64F);
            M.at<double>(0,0) = scale;
            M.at<double>(1,1) = scale;
            warpAffine(image_uchar_orig, image_uchar, M,
                                 cv::Size(DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT),
                                 CV_INTER_CUBIC,
                                 cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
            // resize(image_uchar, image_uchar, Size(new_width, new_height), 0, 0, CV_INTER_CUBIC);
            image_uchar_prev = image_uchar_orig;

            Frame frame;
            frame.scale = scale;
            frame.index = global_counter++;
            frame.video_frame_number = global.uistate.current_frame;
            frame.data_for_wrap = new unsigned char [DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3]; //fill after process
            frame.data_for_mat = new float [DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3];
            process_and_pad_image(frame.data_for_mat, image_uchar, DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT, 0);

            //pad and transform to float
            int offset = 3 * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
            frame.data = new float [BATCH_SIZE * offset];
            int target_width;
            int target_height;
            cv::Mat image_temp;
            for(int i=0; i < BATCH_SIZE; i++) {
                float scale = START_SCALE - i*SCALE_GAP;
                target_width = 16 * ceil(NET_RESOLUTION_WIDTH * scale /16);
                target_height = 16 * ceil(NET_RESOLUTION_HEIGHT * scale /16);

                CHECK_LE(target_width, NET_RESOLUTION_WIDTH);
                CHECK_LE(target_height, NET_RESOLUTION_HEIGHT);

                cv::resize(image_uchar, image_temp, cv::Size(target_width, target_height), 0, 0, CV_INTER_AREA);
                process_and_pad_image(frame.data + i * offset, image_temp, NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT, 1);
            }
            frame.commit_time = get_wall_time();
            frame.preprocessed_time = get_wall_time();
            if(j==0){
                global.input_queue.push(frame);
            }
            else{
                global.input_right_queue.push(frame);
            }
        }
    }
    

    // return nullptr;
}

int connectLimbs(
    std::vector< std::vector<double>> &subset,
    std::vector< std::vector< std::vector<double> > > &connection,
    const float *heatmap_pointer,
    const float *peaks,
    int max_peaks,
    float *joints,
    ModelDescriptor *model_descriptor) {

        const auto num_parts = model_descriptor->get_number_parts();
        const auto limbSeq = model_descriptor->get_limb_sequence();
        const auto mapIdx = model_descriptor->get_map_idx();
        const auto number_limb_seq = model_descriptor->number_limb_sequence();

        int SUBSET_CNT = num_parts+2;
        int SUBSET_SCORE = num_parts+1;
        int SUBSET_SIZE = num_parts+3;

        CHECK_EQ(num_parts, 15);
        CHECK_EQ(number_limb_seq, 14);

        int peaks_offset = 3*(max_peaks+1);
        subset.clear();
        connection.clear();

        for(int k = 0; k < number_limb_seq; k++) {
            const float* map_x = heatmap_pointer + mapIdx[2*k] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
            const float* map_y = heatmap_pointer + mapIdx[2*k+1] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;

            const float* candA = peaks + limbSeq[2*k]*peaks_offset;
            const float* candB = peaks + limbSeq[2*k+1]*peaks_offset;

            std::vector< std::vector<double> > connection_k;
            int nA = candA[0];
            int nB = candB[0];

            // add parts into the subset in special case
            if (nA ==0 && nB ==0) {
                continue;
            }
            else if (nA ==0) {
                for(int i = 1; i <= nB; i++) {
                    std::vector<double> row_vec(SUBSET_SIZE, 0);
                    row_vec[ limbSeq[2*k+1] ] = limbSeq[2*k+1]*peaks_offset + i*3 + 2; //store the index
                    row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
                    row_vec[SUBSET_SCORE] = candB[i*3+2]; //second last number in each row is the total score
                    subset.push_back(row_vec);
                }
                continue;
            }
            else if (nB ==0) {
                for(int i = 1; i <= nA; i++) {
                    std::vector<double> row_vec(SUBSET_SIZE, 0);
                    row_vec[ limbSeq[2*k] ] = limbSeq[2*k]*peaks_offset + i*3 + 2; //store the index
                    row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
                    row_vec[SUBSET_SCORE] = candA[i*3+2]; //second last number in each row is the total score
                    subset.push_back(row_vec);
                }
                continue;
            }

            std::vector< std::vector<double>> temp;
            const int num_inter = 10;

            for(int i = 1; i <= nA; i++) {
                for(int j = 1; j <= nB; j++) {
                    float s_x = candA[i*3];
                    float s_y = candA[i*3+1];
                    float d_x = candB[j*3] - candA[i*3];
                    float d_y = candB[j*3+1] - candA[i*3+1];
                    float norm_vec = sqrt( pow(d_x,2) + pow(d_y,2) );
                    if (norm_vec<1e-6) {
                        continue;
                    }
                    float vec_x = d_x/norm_vec;
                    float vec_y = d_y/norm_vec;

                    float sum = 0;
                    int count = 0;

                    for(int lm=0; lm < num_inter; lm++) {
                        int my = round(s_y + lm*d_y/num_inter);
                        int mx = round(s_x + lm*d_x/num_inter);
                        int idx = my * NET_RESOLUTION_WIDTH + mx;
                        float score = (vec_x*map_x[idx] + vec_y*map_y[idx]);
                        if (score > global.connect_inter_threshold) {
                            sum = sum + score;
                            count ++;
                        }
                    }
                    //float score = sum / count; // + std::min((130/dist-1),0.f)

                    if (count > global.connect_inter_min_above_threshold) {//num_inter*0.8) { //thre/2
                        // parts score + cpnnection score
                        std::vector<double> row_vec(4, 0);
                        row_vec[3] = sum/count + candA[i*3+2] + candB[j*3+2]; //score_all
                        row_vec[2] = sum/count;
                        row_vec[0] = i;
                        row_vec[1] = j;
                        temp.push_back(row_vec);
                    }
                }
            }

        //** select the top num connection, assuming that each part occur only once
        // sort rows in descending order based on parts + connection score
        if (temp.size() > 0)
            std::sort(temp.begin(), temp.end(), ColumnCompare());

        int num = std::min(nA, nB);
        int cnt = 0;
        std::vector<int> occurA(nA, 0);
        std::vector<int> occurB(nB, 0);

        for(int row =0; row < temp.size(); row++) {
            if (cnt==num) {
                break;
            }
            else{
                int i = int(temp[row][0]);
                int j = int(temp[row][1]);
                float score = temp[row][2];
                if ( occurA[i-1] == 0 && occurB[j-1] == 0 ) { // && score> (1+thre)
                    std::vector<double> row_vec(3, 0);
                    row_vec[0] = limbSeq[2*k]*peaks_offset + i*3 + 2;
                    row_vec[1] = limbSeq[2*k+1]*peaks_offset + j*3 + 2;
                    row_vec[2] = score;
                    connection_k.push_back(row_vec);
                    cnt = cnt+1;
                    occurA[i-1] = 1;
                    occurB[j-1] = 1;
                }
            }
        }

        if (k==0) {
            std::vector<double> row_vec(num_parts+3, 0);
            for(int i = 0; i < connection_k.size(); i++) {
                double indexA = connection_k[i][0];
                double indexB = connection_k[i][1];
                row_vec[limbSeq[0]] = indexA;
                row_vec[limbSeq[1]] = indexB;
                row_vec[SUBSET_CNT] = 2;
                // add the score of parts and the connection
                row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
                subset.push_back(row_vec);
            }
        }
        else{
            if (connection_k.size()==0) {
                continue;
            }
            // A is already in the subset, find its connection B
            for(int i = 0; i < connection_k.size(); i++) {
                int num = 0;
                double indexA = connection_k[i][0];
                double indexB = connection_k[i][1];

                for(int j = 0; j < subset.size(); j++) {
                    if (subset[j][limbSeq[2*k]] == indexA) {
                        subset[j][limbSeq[2*k+1]] = indexB;
                        num = num+1;
                        subset[j][SUBSET_CNT] = subset[j][SUBSET_CNT] + 1;
                        subset[j][SUBSET_SCORE] = subset[j][SUBSET_SCORE] + peaks[int(indexB)] + connection_k[i][2];
                    }
                }
                // if can not find partA in the subset, create a new subset
                if (num==0) {
                    std::vector<double> row_vec(SUBSET_SIZE, 0);
                    row_vec[limbSeq[2*k]] = indexA;
                    row_vec[limbSeq[2*k+1]] = indexB;
                    row_vec[SUBSET_CNT] = 2;
                    row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
                    subset.push_back(row_vec);
                }
            }
        }
    }

    //** joints by deleting some rows of subset which has few parts occur
    int cnt = 0;
    for(int i = 0; i < subset.size(); i++) {
        if (subset[i][SUBSET_CNT]>=global.connect_min_subset_cnt && (subset[i][SUBSET_SCORE]/subset[i][SUBSET_CNT])>global.connect_min_subset_score) {
            for(int j = 0; j < num_parts; j++) {
                int idx = int(subset[i][j]);
                if (idx) {
                    joints[cnt*num_parts*3 + j*3 +2] = peaks[idx];
                    joints[cnt*num_parts*3 + j*3 +1] = peaks[idx-1] * DISPLAY_RESOLUTION_HEIGHT/ (float)NET_RESOLUTION_HEIGHT;
                    joints[cnt*num_parts*3 + j*3] = peaks[idx-2] * DISPLAY_RESOLUTION_WIDTH/ (float)NET_RESOLUTION_WIDTH;
                }
                else{
                    joints[cnt*num_parts*3 + j*3 +2] = 0;
                    joints[cnt*num_parts*3 + j*3 +1] = 0;
                    joints[cnt*num_parts*3 + j*3] = 0;
                }
            }
            cnt++;
            if (cnt==MAX_PEOPLE) break;
        }
    }

    return cnt;
}

int distanceThresholdPeaks(const float *in_peaks, int max_peaks,
    float *peaks, ModelDescriptor *model_descriptor) {
    // Post-process peaks to remove those which are within sqrt(dist_threshold2)
    // of each other.

    const auto num_parts = model_descriptor->get_number_parts();
    const float dist_threshold2 = 6*6;
    int peaks_offset = 3*(max_peaks+1);

    int total_peaks = 0;
    for(int p = 0; p < num_parts; p++) {
        const float *pipeaks = in_peaks + p*peaks_offset;
        float *popeaks = peaks + p*peaks_offset;
        int num_in_peaks = int(pipeaks[0]);
        int num_out_peaks = 0; // Actual number of peak count
        for (int c1=0;c1<num_in_peaks;c1++) {
            float x1 = pipeaks[(c1+1)*3+0];
            float y1 = pipeaks[(c1+1)*3+1];
            float s1 = pipeaks[(c1+1)*3+2];
            bool keep = true;
            for (int c2=0;c2<num_out_peaks;c2++) {
                float x2 = popeaks[(c2+1)*3+0];
                float y2 = popeaks[(c2+1)*3+1];
                float s2 = popeaks[(c2+1)*3+2];
                float dist2 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
                if (dist2<dist_threshold2) {
                    // This peak is too close to a peak already in the output buffer
                    // so don't add it.
                    keep = false;
                    if (s1>s2) {
                        // It's better than the one in the output buffer
                        // so we swap it.
                        popeaks[(c2+1)*3+0] = x1;
                        popeaks[(c2+1)*3+1] = y1;
                        popeaks[(c2+1)*3+2] = s1;
                    }
                }
            }
            if (keep && num_out_peaks<max_peaks) {
                // We don't already have a better peak within the threshold distance
                popeaks[(num_out_peaks+1)*3+0] = x1;
                popeaks[(num_out_peaks+1)*3+1] = y1;
                popeaks[(num_out_peaks+1)*3+2] = s1;
                num_out_peaks++;
            }
        }
        // if (num_in_peaks!=num_out_peaks) {
            //LOG(INFO) << "Part: " << p << " in peaks: "<< num_in_peaks << " out: " << num_out_peaks;
        // }
        popeaks[0] = float(num_out_peaks);
        total_peaks += num_out_peaks;
    }
    return total_peaks;
}

int connectLimbsCOCO(
    std::vector< std::vector<double>> &subset,
    std::vector< std::vector< std::vector<double> > > &connection,
    const float *heatmap_pointer,
    const float *in_peaks,
    int max_peaks,
    float *joints,
    ModelDescriptor *model_descriptor) {
        /* Parts Connection ---------------------------------------*/
        const auto num_parts = model_descriptor->get_number_parts();
        const auto limbSeq = model_descriptor->get_limb_sequence();
        const auto mapIdx = model_descriptor->get_map_idx();
        const auto number_limb_seq = model_descriptor->number_limb_sequence();

        CHECK_EQ(num_parts, 18) << "Wrong connection function for model";
        CHECK_EQ(number_limb_seq, 19) << "Wrong connection function for model";

        int SUBSET_CNT = num_parts+2;
        int SUBSET_SCORE = num_parts+1;
        int SUBSET_SIZE = num_parts+3;

        const int peaks_offset = 3*(max_peaks+1);

        const float *peaks = in_peaks;
        subset.clear();
        connection.clear();

        for(int k = 0; k < number_limb_seq; k++) {
            const float* map_x = heatmap_pointer + mapIdx[2*k] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
            const float* map_y = heatmap_pointer + mapIdx[2*k+1] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;

            const float* candA = peaks + limbSeq[2*k]*peaks_offset;
            const float* candB = peaks + limbSeq[2*k+1]*peaks_offset;

            std::vector< std::vector<double> > connection_k;
            int nA = candA[0];
            int nB = candB[0];

            // add parts into the subset in special case
            if (nA ==0 && nB ==0) {
                continue;
            } else if (nA ==0) {
                for(int i = 1; i <= nB; i++) {
                    int num = 0;
                    int indexB = limbSeq[2*k+1];
                    for(int j = 0; j < subset.size(); j++) {
                            int off = limbSeq[2*k+1]*peaks_offset + i*3 + 2;
                            if (subset[j][indexB] == off) {
                                    num = num+1;
                                    continue;
                            }
                    }
                    if (num!=0) {
                        //LOG(INFO) << " else if (nA==0) shouldn't have any nB already assigned?";
                    } else {
                        std::vector<double> row_vec(SUBSET_SIZE, 0);
                        row_vec[ limbSeq[2*k+1] ] = limbSeq[2*k+1]*peaks_offset + i*3 + 2; //store the index
                        row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
                        row_vec[SUBSET_SCORE] = candB[i*3+2]; //second last number in each row is the total score
                        subset.push_back(row_vec);
                    }
                    //LOG(INFO) << "nA==0 New subset on part " << k << " subsets: " << subset.size();
                }
                continue;
            } else if (nB ==0) {
                for(int i = 1; i <= nA; i++) {
                    int num = 0;
                    int indexA = limbSeq[2*k];
                    for(int j = 0; j < subset.size(); j++) {
                            int off = limbSeq[2*k]*peaks_offset + i*3 + 2;
                            if (subset[j][indexA] == off) {
                                    num = num+1;
                                    continue;
                            }
                    }
                    if (num==0) {
                        std::vector<double> row_vec(SUBSET_SIZE, 0);
                        row_vec[ limbSeq[2*k] ] = limbSeq[2*k]*peaks_offset + i*3 + 2; //store the index
                        row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
                        row_vec[SUBSET_SCORE] = candA[i*3+2]; //second last number in each row is the total score
                        subset.push_back(row_vec);
                        //LOG(INFO) << "nB==0 New subset on part " << k << " subsets: " << subset.size();
                    } else {
                        //LOG(INFO) << "nB==0 discarded would have added";
                    }
                }
                continue;
            }

            std::vector< std::vector<double>> temp;
            const int num_inter = 10;

            for(int i = 1; i <= nA; i++) {
                for(int j = 1; j <= nB; j++) {
                    float s_x = candA[i*3];
                    float s_y = candA[i*3+1];
                    float d_x = candB[j*3] - candA[i*3];
                    float d_y = candB[j*3+1] - candA[i*3+1];
                    float norm_vec = sqrt( d_x*d_x + d_y*d_y );
                    if (norm_vec<1e-6) {
                        // The peaks are coincident. Don't connect them.
                        continue;
                    }
                    float vec_x = d_x/norm_vec;
                    float vec_y = d_y/norm_vec;

                    float sum = 0;
                    int count = 0;

                    for(int lm=0; lm < num_inter; lm++) {
                        int my = round(s_y + lm*d_y/num_inter);
                        int mx = round(s_x + lm*d_x/num_inter);
                        if (mx>=NET_RESOLUTION_WIDTH) {
                            //LOG(ERROR) << "mx " << mx << "out of range";
                            mx = NET_RESOLUTION_WIDTH-1;
                        }
                        if (my>=NET_RESOLUTION_HEIGHT) {
                            //LOG(ERROR) << "my " << my << "out of range";
                            my = NET_RESOLUTION_HEIGHT-1;
                        }
                        CHECK_GE(mx,0);
                        CHECK_GE(my,0);
                        int idx = my * NET_RESOLUTION_WIDTH + mx;
                        float score = (vec_x*map_x[idx] + vec_y*map_y[idx]);
                        if (score > global.connect_inter_threshold) {
                            sum = sum + score;
                            count ++;
                        }
                    }
                    //float score = sum / count; // + std::min((130/dist-1),0.f)

                    if (count > global.connect_inter_min_above_threshold) {//num_inter*0.8) { //thre/2
                        // parts score + cpnnection score
                        std::vector<double> row_vec(4, 0);
                        row_vec[3] = sum/count + candA[i*3+2] + candB[j*3+2]; //score_all
                        row_vec[2] = sum/count;
                        row_vec[0] = i;
                        row_vec[1] = j;
                        temp.push_back(row_vec);
                    }
                }
            }

            //** select the top num connection, assuming that each part occur only once
            // sort rows in descending order based on parts + connection score
            if (temp.size() > 0)
                std::sort(temp.begin(), temp.end(), ColumnCompare());

            int num = std::min(nA, nB);
            int cnt = 0;
            std::vector<int> occurA(nA, 0);
            std::vector<int> occurB(nB, 0);

            for(int row =0; row < temp.size(); row++) {
                if (cnt==num) {
                    break;
                }
                else{
                    int i = int(temp[row][0]);
                    int j = int(temp[row][1]);
                    float score = temp[row][2];
                    if ( occurA[i-1] == 0 && occurB[j-1] == 0 ) { // && score> (1+thre)
                        std::vector<double> row_vec(3, 0);
                        row_vec[0] = limbSeq[2*k]*peaks_offset + i*3 + 2;
                        row_vec[1] = limbSeq[2*k+1]*peaks_offset + j*3 + 2;
                        row_vec[2] = score;
                        connection_k.push_back(row_vec);
                        cnt = cnt+1;
                        occurA[i-1] = 1;
                        occurB[j-1] = 1;
                    }
                }
            }

            //** cluster all the joints candidates into subset based on the part connection
            // initialize first body part connection 15&16
            if (k==0) {
                std::vector<double> row_vec(num_parts+3, 0);
                for(int i = 0; i < connection_k.size(); i++) {
                    double indexB = connection_k[i][1];
                    double indexA = connection_k[i][0];
                    row_vec[limbSeq[0]] = indexA;
                    row_vec[limbSeq[1]] = indexB;
                    row_vec[SUBSET_CNT] = 2;
                    // add the score of parts and the connection
                    row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
                    //LOG(INFO) << "New subset on part " << k << " subsets: " << subset.size();
                    subset.push_back(row_vec);
                }
            }/* else if (k==17 || k==18) { // TODO: Check k numbers?
                //   %add 15 16 connection
                for(int i = 0; i < connection_k.size(); i++) {
                    double indexA = connection_k[i][0];
                    double indexB = connection_k[i][1];

                    for(int j = 0; j < subset.size(); j++) {
                    // if subset(j, indexA) == partA(i) && subset(j, indexB) == 0
                    //         subset(j, indexB) = partB(i);
                    // elseif subset(j, indexB) == partB(i) && subset(j, indexA) == 0
                    //         subset(j, indexA) = partA(i);
                    // end
                        if (subset[j][limbSeq[2*k]] == indexA && subset[j][limbSeq[2*k+1]]==0) {
                            subset[j][limbSeq[2*k+1]] = indexB;
                        } else if (subset[j][limbSeq[2*k+1]] == indexB && subset[j][limbSeq[2*k]]==0) {
                            subset[j][limbSeq[2*k]] = indexA;
                        }
                }
                continue;
            }
        }*/ else{
            if (connection_k.size()==0) {
                continue;
            }

            // A is already in the subset, find its connection B
            for(int i = 0; i < connection_k.size(); i++) {
                int num = 0;
                double indexA = connection_k[i][0];
                double indexB = connection_k[i][1];

                for(int j = 0; j < subset.size(); j++) {
                    if (subset[j][limbSeq[2*k]] == indexA) {
                        subset[j][limbSeq[2*k+1]] = indexB;
                        num = num+1;
                        subset[j][SUBSET_CNT] = subset[j][SUBSET_CNT] + 1;
                        subset[j][SUBSET_SCORE] = subset[j][SUBSET_SCORE] + peaks[int(indexB)] + connection_k[i][2];
                    }
                }
                // if can not find partA in the subset, create a new subset
                if (num==0) {
                    //LOG(INFO) << "New subset on part " << k << " subsets: " << subset.size();
                    std::vector<double> row_vec(SUBSET_SIZE, 0);
                    row_vec[limbSeq[2*k]] = indexA;
                    row_vec[limbSeq[2*k+1]] = indexB;
                    row_vec[SUBSET_CNT] = 2;
                    row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
                    subset.push_back(row_vec);
                }
            }
        }
    }

    //** joints by deleteing some rows of subset which has few parts occur
    int cnt = 0;
    for(int i = 0; i < subset.size(); i++) {
        if (subset[i][SUBSET_CNT]<1) {
            LOG(INFO) << "BAD SUBSET_CNT";
        }
        if (subset[i][SUBSET_CNT]>=global.connect_min_subset_cnt && (subset[i][SUBSET_SCORE]/subset[i][SUBSET_CNT])>global.connect_min_subset_score) {
            for(int j = 0; j < num_parts; j++) {
                int idx = int(subset[i][j]);
                if (idx) {
                    joints[cnt*num_parts*3 + j*3 +2] = peaks[idx];
                    joints[cnt*num_parts*3 + j*3 +1] = peaks[idx-1]* DISPLAY_RESOLUTION_HEIGHT/ (float)NET_RESOLUTION_HEIGHT;//(peaks[idx-1] - padh) * ratio_h;
                    joints[cnt*num_parts*3 + j*3] = peaks[idx-2]* DISPLAY_RESOLUTION_WIDTH/ (float)NET_RESOLUTION_WIDTH;//(peaks[idx-2] -padw) * ratio_w;
                }
                else{
                    joints[cnt*num_parts*3 + j*3 +2] = 0;
                    joints[cnt*num_parts*3 + j*3 +1] = 0;
                    joints[cnt*num_parts*3 + j*3] = 0;
                }
            }
            cnt++;
            if (cnt==MAX_PEOPLE) break;
        }
    }

    return cnt;
}


void* processLeftFrame(void *i) {
    int tid = *((int *) i);
    ROS_INFO("processLeftFrame tid is %i", tid);
    warmup(tid);
    ROS_INFO("processLeftFrame warmed up. left_processing is %s", left_processing ? "true" : "false");
    LOG(INFO) << "GPU " << tid << " is ready";

    Frame frame;

    int offset = NET_RESOLUTION_WIDTH * NET_RESOLUTION_HEIGHT * 3;
    //bool empty = false;

    Frame frame_batch[BATCH_SIZE];

    std::vector< std::vector<double>> subset;
    std::vector< std::vector< std::vector<double> > > connection;

    const boost::shared_ptr<caffe::Blob<float>> heatmap_blob = net_copies[tid].person_net->blob_by_name("resized_map");
    const boost::shared_ptr<caffe::Blob<float>> joints_blob = net_copies[tid].person_net->blob_by_name("joints");

    caffe::NmsLayer<float> *nms_layer = (caffe::NmsLayer<float>*)net_copies[tid].person_net->layer_by_name("nms").get();
    // printf("check point 6\n");
    left_processing = false;
    //while(!empty) {
    while(1) {

        if (global.quit_threads)
            break;

        //LOG(ERROR) << "start";
        int valid_data = 0;
        //for(int n = 0; n < BATCH_SIZE; n++) {
        while(valid_data<1) {
            if (global.input_queue.try_pop(&frame)) {
                //consider dropping it
                frame.gpu_fetched_time = get_wall_time();
                double elaspsed_time = frame.gpu_fetched_time - frame.commit_time;
                //LOG(ERROR) << "frame " << frame.index << " is copied to GPU after " << elaspsed_time << " sec";
                if (elaspsed_time > 0.1
                   && !FLAGS_no_frame_drops) {//0.1*BATCH_SIZE) { //0.1*BATCH_SIZE
                    //drop frame
                    VLOG(1) << "skip frame " << frame.index;
                    delete [] frame.data;
                    delete [] frame.data_for_mat;
                    delete [] frame.data_for_wrap;
                    //n--;

                    const std::lock_guard<std::mutex> lock{global.mutex};
                    global.dropped_index.push(frame.index);
                    continue;
                }
                //double tic1  = get_wall_time();
                // printf("Before cudaMemcpy;\n");
                cudaMemcpy(net_copies[tid].canvas, frame.data_for_mat, DISPLAY_RESOLUTION_WIDTH * DISPLAY_RESOLUTION_HEIGHT * 3 * sizeof(float), cudaMemcpyHostToDevice);
                // printf("After cudaMemcpy;\n");
                frame_batch[0] = frame;
                //LOG(ERROR)<< "Copy data " << index_array[n] << " to device " << tid << ", now size " << global.input_queue.size();
                float* pointer = net_copies[tid].person_net->blobs()[0]->mutable_gpu_data();

                cudaMemcpy(pointer + 0 * offset, frame_batch[0].data, BATCH_SIZE * offset * sizeof(float), cudaMemcpyHostToDevice);
                valid_data++;
                //VLOG(2) << "Host->device " << (get_wall_time()-tic1)*1000.0 << " ms.";
            }
            else {
                //empty = true;
                break;
            }
        }
        // printf("check point 6 looping\n");
        if (valid_data == 0)
            continue;

        // printf("check point 7\n");
        nms_layer->SetThreshold(global.nms_threshold);
        net_copies[tid].person_net->ForwardFrom(0);
        VLOG(2) << "CNN time " << (get_wall_time()-frame.gpu_fetched_time)*1000.0 << " ms.";
        //cudaDeviceSynchronize();
        float* heatmap_pointer = heatmap_blob->mutable_cpu_data();
        const float* peaks = joints_blob->mutable_cpu_data();

        float joints[MAX_NUM_PARTS*3*MAX_PEOPLE]; //10*15*3

        int cnt = 0;
        // CHECK_EQ(net_copies[tid].nms_num_parts, 15);
        double tic = get_wall_time();
        const int num_parts = net_copies[tid].nms_num_parts;
        if (net_copies[tid].nms_num_parts==15) {
            cnt = connectLimbs(subset, connection,
                                                 heatmap_pointer, peaks,
                                                 net_copies[tid].nms_max_peaks, joints, net_copies[tid].up_model_descriptor.get());
        } else {
            cnt = connectLimbsCOCO(subset, connection,
                                                 heatmap_pointer, peaks,
                                                 net_copies[tid].nms_max_peaks, joints, net_copies[tid].up_model_descriptor.get());
        }

        VLOG(2) << "CNT: " << cnt << " Connect time " << (get_wall_time()-tic)*1000.0 << " ms.";
        net_copies[tid].num_people[0] = cnt;
        VLOG(2) << "num_people[i] = " << cnt;


        cudaMemcpy(net_copies[tid].joints, joints,
            MAX_NUM_PARTS*3*MAX_PEOPLE * sizeof(float),
            cudaMemcpyHostToDevice);
        // printf("check point 8\n");
        if (subset.size() != 0) {
            //LOG(ERROR) << "Rendering";
            render(tid, heatmap_pointer); //only support batch size = 1!!!!
            for(int n = 0; n < valid_data; n++) {
                frame_batch[n].numPeople = net_copies[tid].num_people[n];
                frame_batch[n].gpu_computed_time = get_wall_time();
                frame_batch[n].joints = boost::shared_ptr<float[]>(new float[frame_batch[n].numPeople*MAX_NUM_PARTS*3]);
                for (int ij=0;ij<frame_batch[n].numPeople*num_parts*3;ij++) {
                    frame_batch[n].joints[ij] = joints[ij];
                }


                cudaMemcpy(frame_batch[n].data_for_mat, net_copies[tid].canvas, DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3 * sizeof(float), cudaMemcpyDeviceToHost);
                global.output_queue.push(frame_batch[n]);
            }
        }
        else {
            render(tid, heatmap_pointer);
            //frame_batch[n].data should revert to 0-255
            for(int n = 0; n < valid_data; n++) {
                frame_batch[n].numPeople = 0;
                frame_batch[n].gpu_computed_time = get_wall_time();
                cudaMemcpy(frame_batch[n].data_for_mat, net_copies[tid].canvas, DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3 * sizeof(float), cudaMemcpyDeviceToHost);
                global.output_queue.push(frame_batch[n]);
            }
        }
        left_processing = false;
        // printf("check point 10\n");
    }
    return nullptr;
}

void* processRightFrame(void *i) {
    int tid = *((int *) i);
    ROS_INFO("processRightFrame tid is %i", tid);
    warmup(tid);
    ROS_INFO("processRightFrame warmed up. right_processing is %s", right_processing ? "true" : "false");
    LOG(INFO) << "GPU " << tid << " is ready";

    Frame frame;

    int offset = NET_RESOLUTION_WIDTH * NET_RESOLUTION_HEIGHT * 3;
    //bool empty = false;

    Frame frame_batch[BATCH_SIZE];

    std::vector< std::vector<double>> subset;
    std::vector< std::vector< std::vector<double> > > connection;

    const boost::shared_ptr<caffe::Blob<float>> heatmap_blob = net_copies[tid].person_net->blob_by_name("resized_map");
    const boost::shared_ptr<caffe::Blob<float>> joints_blob = net_copies[tid].person_net->blob_by_name("joints");

    caffe::NmsLayer<float> *nms_layer = (caffe::NmsLayer<float>*)net_copies[tid].person_net->layer_by_name("nms").get();
    // printf("check point 6\n");
    right_processing = false;
    //while(!empty) {
    while(1) {

        if (global.quit_threads)
            break;

        //LOG(ERROR) << "start";
        int valid_data = 0;
        //for(int n = 0; n < BATCH_SIZE; n++) {
        while(valid_data<1) {
            if (global.input_right_queue.try_pop(&frame)) {
                //consider dropping it
                frame.gpu_fetched_time = get_wall_time();
                double elaspsed_time = frame.gpu_fetched_time - frame.commit_time;
                //LOG(ERROR) << "frame " << frame.index << " is copied to GPU after " << elaspsed_time << " sec";
                if (elaspsed_time > 0.1
                   && !FLAGS_no_frame_drops) {//0.1*BATCH_SIZE) { //0.1*BATCH_SIZE
                    //drop frame
                    VLOG(1) << "skip frame " << frame.index;
                    delete [] frame.data;
                    delete [] frame.data_for_mat;
                    delete [] frame.data_for_wrap;
                    //n--;

                    const std::lock_guard<std::mutex> lock{global.mutex};
                    global.dropped_index.push(frame.index);
                    continue;
                }
                //double tic1  = get_wall_time();
                // printf("Before cudaMemcpy;\n");
                cudaMemcpy(net_copies[tid].canvas, frame.data_for_mat, DISPLAY_RESOLUTION_WIDTH * DISPLAY_RESOLUTION_HEIGHT * 3 * sizeof(float), cudaMemcpyHostToDevice);
                // printf("After cudaMemcpy;\n");
                frame_batch[0] = frame;
                //LOG(ERROR)<< "Copy data " << index_array[n] << " to device " << tid << ", now size " << global.input_queue.size();
                float* pointer = net_copies[tid].person_net->blobs()[0]->mutable_gpu_data();

                cudaMemcpy(pointer + 0 * offset, frame_batch[0].data, BATCH_SIZE * offset * sizeof(float), cudaMemcpyHostToDevice);
                valid_data++;
                //VLOG(2) << "Host->device " << (get_wall_time()-tic1)*1000.0 << " ms.";
            }
            else {
                //empty = true;
                break;
            }
        }
        // printf("check point 6 looping\n");
        if (valid_data == 0)
            continue;

        // printf("check point 7\n");
        nms_layer->SetThreshold(global.nms_threshold);
        net_copies[tid].person_net->ForwardFrom(0);
        VLOG(2) << "CNN time " << (get_wall_time()-frame.gpu_fetched_time)*1000.0 << " ms.";
        //cudaDeviceSynchronize();
        float* heatmap_pointer = heatmap_blob->mutable_cpu_data();
        const float* peaks = joints_blob->mutable_cpu_data();

        float joints[MAX_NUM_PARTS*3*MAX_PEOPLE]; //10*15*3

        int cnt = 0;
        // CHECK_EQ(net_copies[tid].nms_num_parts, 15);
        double tic = get_wall_time();
        const int num_parts = net_copies[tid].nms_num_parts;
        if (net_copies[tid].nms_num_parts==15) {
            cnt = connectLimbs(subset, connection,
                                                 heatmap_pointer, peaks,
                                                 net_copies[tid].nms_max_peaks, joints, net_copies[tid].up_model_descriptor.get());
        } else {
            cnt = connectLimbsCOCO(subset, connection,
                                                 heatmap_pointer, peaks,
                                                 net_copies[tid].nms_max_peaks, joints, net_copies[tid].up_model_descriptor.get());
        }

        VLOG(2) << "CNT: " << cnt << " Connect time " << (get_wall_time()-tic)*1000.0 << " ms.";
        net_copies[tid].num_people[0] = cnt;
        VLOG(2) << "num_people[i] = " << cnt;


        cudaMemcpy(net_copies[tid].joints, joints,
            MAX_NUM_PARTS*3*MAX_PEOPLE * sizeof(float),
            cudaMemcpyHostToDevice);
        // printf("check point 8\n");
        if (subset.size() != 0) {
            //LOG(ERROR) << "Rendering";
            render(tid, heatmap_pointer); //only support batch size = 1!!!!
            for(int n = 0; n < valid_data; n++) {
                frame_batch[n].numPeople = net_copies[tid].num_people[n];
                frame_batch[n].gpu_computed_time = get_wall_time();
                frame_batch[n].joints = boost::shared_ptr<float[]>(new float[frame_batch[n].numPeople*MAX_NUM_PARTS*3]);
                for (int ij=0;ij<frame_batch[n].numPeople*num_parts*3;ij++) {
                    frame_batch[n].joints[ij] = joints[ij];
                }


                cudaMemcpy(frame_batch[n].data_for_mat, net_copies[tid].canvas, DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3 * sizeof(float), cudaMemcpyDeviceToHost);
                global.output_right_queue.push(frame_batch[n]);
            }
        }
        else {
            render(tid, heatmap_pointer);
            //frame_batch[n].data should revert to 0-255
            for(int n = 0; n < valid_data; n++) {
                frame_batch[n].numPeople = 0;
                frame_batch[n].gpu_computed_time = get_wall_time();
                cudaMemcpy(frame_batch[n].data_for_mat, net_copies[tid].canvas, DISPLAY_RESOLUTION_HEIGHT * DISPLAY_RESOLUTION_WIDTH * 3 * sizeof(float), cudaMemcpyDeviceToHost);
                global.output_right_queue.push(frame_batch[n]);
            }
        }
        right_processing = false;
        // printf("check point 10\n");
    }
    return nullptr;
}

class FrameCompare{
public:
    bool operator() (const Frame &a, const Frame &b) const{
        return a.index > b.index;
    }
};

void* buffer_and_order_left(void* threadargs) { //only one thread can execute this
    FrameCompare comp;
    std::priority_queue<Frame, std::vector<Frame>, FrameCompare> buffer(comp);
    Frame frame;

    int frame_waited = 1;
    while(1) {
        if (global.quit_threads)
            break;
        bool success = global.output_queue_mated.try_pop(&frame);
        frame.buffer_start_time = get_wall_time();
        if (success) {
            frame.buffer_end_time = get_wall_time();
            global.output_queue_ordered.push(frame);
        }
        else {
            //output_queue
        }
    }
    return nullptr;
}

void* buffer_and_order_right(void* threadargs) { //only one thread can execute this
    FrameCompare comp;
    std::priority_queue<Frame, std::vector<Frame>, FrameCompare> buffer(comp);
    Frame frame;

    int frame_waited = 1;
    while(1) {
        if (global.quit_threads)
            break;
        bool success = global.output_right_queue_mated.try_pop(&frame);
        frame.buffer_start_time = get_wall_time();
        if (success) {
            frame.buffer_end_time = get_wall_time();
            global.output_right_queue_ordered.push(frame);
        }
        else {
            //output_queue
        }
    }
    return nullptr;
}

void* postProcessLeftFrame(void *i) {
    //int tid = *((int *) i);
    Frame frame;

    while(1) {
        if (global.quit_threads)
            break;

        frame = global.output_queue.pop();
        frame.postprocesse_begin_time = get_wall_time();

        //Mat visualize(NET_RESOLUTION_HEIGHT, NET_RESOLUTION_WIDTH, CV_8UC3);
        int offset = DISPLAY_RESOLUTION_WIDTH * DISPLAY_RESOLUTION_HEIGHT;
        for(int c = 0; c < 3; c++) {
            for(int i = 0; i < DISPLAY_RESOLUTION_HEIGHT; i++) {
                for(int j = 0; j < DISPLAY_RESOLUTION_WIDTH; j++) {
                    int value = int(frame.data_for_mat[c*offset + i*DISPLAY_RESOLUTION_WIDTH + j] + 0.5);
                    value = value<0 ? 0 : (value > 255 ? 255 : value);
                    frame.data_for_wrap[3*(i*DISPLAY_RESOLUTION_WIDTH + j) + c] = (unsigned char)(value);
                }
            }
        }
        frame.postprocesse_end_time = get_wall_time();
        global.output_queue_mated.push(frame);

    }
    return nullptr;
}

void* postProcessRightFrame(void *i) {
    //int tid = *((int *) i);
    Frame frame;

    while(1) {
        if (global.quit_threads)
            break;

        frame = global.output_right_queue.pop();
        frame.postprocesse_begin_time = get_wall_time();

        //Mat visualize(NET_RESOLUTION_HEIGHT, NET_RESOLUTION_WIDTH, CV_8UC3);
        int offset = DISPLAY_RESOLUTION_WIDTH * DISPLAY_RESOLUTION_HEIGHT;
        for(int c = 0; c < 3; c++) {
            for(int i = 0; i < DISPLAY_RESOLUTION_HEIGHT; i++) {
                for(int j = 0; j < DISPLAY_RESOLUTION_WIDTH; j++) {
                    int value = int(frame.data_for_mat[c*offset + i*DISPLAY_RESOLUTION_WIDTH + j] + 0.5);
                    value = value<0 ? 0 : (value > 255 ? 255 : value);
                    frame.data_for_wrap[3*(i*DISPLAY_RESOLUTION_WIDTH + j) + c] = (unsigned char)(value);
                }
            }
        }
        frame.postprocesse_end_time = get_wall_time();
        global.output_right_queue_mated.push(frame);

    }
    return nullptr;
}

void* displayFrame(void *i) { //single thread
    Frame frame;
    int counter = 1;
    double last_time = get_wall_time();
    double this_time;
      float FPS = 0;
    char tmp_str[256];
    while(1) {
        if (global.quit_threads)
            break;

        frame = global.output_queue_ordered.pop();
        double tic = get_wall_time();
        cv::Mat wrap_frame(DISPLAY_RESOLUTION_HEIGHT, DISPLAY_RESOLUTION_WIDTH, CV_8UC3, frame.data_for_wrap);

        if (FLAGS_write_frames.empty()) {
            snprintf(tmp_str, 256, "%4.1f fps", FPS);
        } else {
            snprintf(tmp_str, 256, "%4.2f s/gpu", FLAGS_num_gpu*1.0/FPS);
        }
        if (1) {
        cv::putText(wrap_frame, tmp_str, cv::Point(25,35),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255,150,150), 1);

        snprintf(tmp_str, 256, "%4d", frame.numPeople);
        cv::putText(wrap_frame, tmp_str, cv::Point(DISPLAY_RESOLUTION_WIDTH-100+2, 35+2),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
        cv::putText(wrap_frame, tmp_str, cv::Point(DISPLAY_RESOLUTION_WIDTH-100, 35),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
        }
        if (global.part_to_show!=0) {
            if (global.part_to_show-1<=net_copies.at(0).up_model_descriptor->get_number_parts()) {
                snprintf(tmp_str, 256, "%10s", net_copies.at(0).up_model_descriptor->get_part_name(global.part_to_show-1).c_str());
            } else {
                int aff_part = ((global.part_to_show-1)-net_copies.at(0).up_model_descriptor->get_number_parts()-1)*2;
                if (aff_part==0) {
                    snprintf(tmp_str, 256, "%10s", "PAFs");
                } else {
                    aff_part = aff_part-2;
                    aff_part += 1+net_copies.at(0).up_model_descriptor->get_number_parts();
                    std::string uvname = net_copies.at(0).up_model_descriptor->get_part_name(aff_part);
                    std::string conn = uvname.substr(0, uvname.find("("));
                    snprintf(tmp_str, 256, "%10s", conn.c_str());
                }
            }
            cv::putText(wrap_frame, tmp_str, cv::Point(DISPLAY_RESOLUTION_WIDTH-175+1, 55+1),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
        }
        if (!FLAGS_video.empty() && FLAGS_write_frames.empty()) {
            snprintf(tmp_str, 256, "Frame %6d", global.uistate.current_frame);
            // cv::putText(wrap_frame, tmp_str, cv::Point(27,37),
            //     cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 2);
            cv::putText(wrap_frame, tmp_str, cv::Point(25,55),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255,255,255), 1);
        }

        if (!FLAGS_no_display) {
            cv::imshow("video", wrap_frame);
        }
        if (!FLAGS_write_frames.empty()) {
            std::vector<int> compression_params;
            compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
            compression_params.push_back(98);
            char fname[256];
            if (FLAGS_image_dir.empty()) {
                sprintf(fname, "%s/frame%06d.jpg", FLAGS_write_frames.c_str(), frame.video_frame_number);
            } else {
                boost::filesystem::path p(global.image_list[frame.video_frame_number]);
                std::string rawname = p.stem().string();
                sprintf(fname, "%s/%s.jpg", FLAGS_write_frames.c_str(), rawname.c_str());
            }

            cv::imwrite(fname, wrap_frame, compression_params);
        }

        if (!FLAGS_write_json.empty()) {
            double scale = 1.0/frame.scale;
            const int num_parts = net_copies.at(0).up_model_descriptor->get_number_parts();
            char fname[256];
            if (FLAGS_image_dir.empty()) {
                sprintf(fname, "%s/frame%06d.json", FLAGS_write_json.c_str(), frame.video_frame_number);
            } else {
                boost::filesystem::path p(global.image_list[frame.video_frame_number]);
                std::string rawname = p.stem().string();

                sprintf(fname, "%s/%s.json", FLAGS_write_json.c_str(), rawname.c_str());
            }
            std::ofstream fs(fname);
            fs << "{\n";
            fs << "\"version\":0.1,\n";
            fs << "\"bodies\":[\n";
            for (int ip=0;ip<frame.numPeople;ip++) {
                fs << "{\n" << "\"joints\":" << "[";
                for (int ij=0;ij<num_parts;ij++) {
                    fs << scale*frame.joints[ip*num_parts*3 + ij*3+0] << ",";
                    fs << scale*frame.joints[ip*num_parts*3 + ij*3+1] << ",";
                    fs << frame.joints[ip*num_parts*3 + ij*3+2];
                    if (ij<num_parts-1) fs << ",";
                }
                fs << "]\n";
                fs << "}";
                if (ip<frame.numPeople-1) {
                    fs<<",\n";
                }
            }
            fs << "]\n";
            fs << "}\n";
            // last_time += get_wall_time()-a;
        }


        counter++;

        if (counter % 30 == 0) {
            this_time = get_wall_time();
            FPS = 30.0f / (this_time - last_time);
            global.uistate.fps = FPS;
            //LOG(ERROR) << frame.cols << "  " << frame.rows;
            last_time = this_time;
            char msg[1000];
            sprintf(msg, "# %d, NP %d, Latency %.3f, Preprocess %.3f, QueueA %.3f, GPU %.3f, QueueB %.3f, Postproc %.3f, QueueC %.3f, Buffered %.3f, QueueD %.3f, FPS = %.1f",
                  frame.index, frame.numPeople,
                  this_time - frame.commit_time,
                  frame.preprocessed_time - frame.commit_time,
                  frame.gpu_fetched_time - frame.preprocessed_time,
                  frame.gpu_computed_time - frame.gpu_fetched_time,
                  frame.postprocesse_begin_time - frame.gpu_computed_time,
                  frame.postprocesse_end_time - frame.postprocesse_begin_time,
                  frame.buffer_start_time - frame.postprocesse_end_time,
                  frame.buffer_end_time - frame.buffer_start_time,
                  this_time - frame.buffer_end_time,
                  FPS);
            LOG(INFO) << msg;
        }

        delete [] frame.data_for_mat;
        delete [] frame.data_for_wrap;
        delete [] frame.data;

        //LOG(ERROR) << msg;
        int key = cv::waitKey(1);
        if (!handleKey(key)) {
            // TODO: sync issues?
            break;
        }

        VLOG(2) << "Display time " << (get_wall_time()-tic)*1000.0 << " ms.";
    }
    return nullptr;
}

void publishStereoImgToROS(const cv::Mat wrap_frame_left, const cv::Mat wrap_frame_right)  {
    sensor_msgs::ImagePtr output_left = cv_bridge::CvImage(std_msgs::Header(), "rgb8", wrap_frame_left).toImageMsg();
    sensor_msgs::ImagePtr output_right = cv_bridge::CvImage(std_msgs::Header(), "rgb8", wrap_frame_right).toImageMsg();
    ros::Time now = ros::Time::now();
    output_left->header.stamp = now;
    output_right->header.stamp = now;
    poseLeftImagePublisher.publish(output_left);
    poseRightImagePublisher.publish(output_right);
}

void displayROSFrameStereo(const sensor_msgs::ImageConstPtr& msg) { //single thread
    Frame frame;
    Frame frame_right;
    double last_time;
    double this_time;
    float FPS;
    FPS = global.uistate.fps;
    char tmp_str[256];
    // std_msgs::Int32MultiArray array;
    std_msgs::Float32MultiArray array;
    //Clear array
    array.data.clear();


    if (global.output_queue_ordered.size()>0 && global.output_right_queue_ordered.size()>0) {
        // ROS_INFO("Left queue size: %lu", global.output_queue_ordered.size());
        // ROS_INFO("Right queue size: %lu", global.output_right_queue_ordered.size());
        // global.output_queue_ordered.try_pop(&frame)
        frame = global.output_queue_ordered.pop();
        frame_right = global.output_right_queue_ordered.pop();
        double tic = get_wall_time();
        cv::Mat wrap_frame(DISPLAY_RESOLUTION_HEIGHT, DISPLAY_RESOLUTION_WIDTH, CV_8UC3, frame.data_for_wrap);
        cv::Mat wrap_frame_right(DISPLAY_RESOLUTION_HEIGHT, DISPLAY_RESOLUTION_WIDTH, CV_8UC3, frame_right.data_for_wrap);

        if (display_counter % 30 == 0) {
            this_time = get_wall_time();
            FPS = 30.0f / (this_time - last_pop_time);
            global.uistate.fps = FPS;
            //LOG(ERROR) << frame.cols << "  " << frame.rows;
            last_time = this_time;
            char msg[1000];
            sprintf(msg, "# %d, NP %d, Latency %.3f, Preprocess %.3f, QueueA %.3f, GPU %.3f, QueueB %.3f, Postproc %.3f, QueueC %.3f, Buffered %.3f, QueueD %.3f, FPS = %.1f",
                  frame.index, frame.numPeople,
                  this_time - frame.commit_time,
                  frame.preprocessed_time - frame.commit_time,
                  frame.gpu_fetched_time - frame.preprocessed_time,
                  frame.gpu_computed_time - frame.gpu_fetched_time,
                  frame.postprocesse_begin_time - frame.gpu_computed_time,
                  frame.postprocesse_end_time - frame.postprocesse_begin_time,
                  frame.buffer_start_time - frame.postprocesse_end_time,
                  frame.buffer_end_time - frame.buffer_start_time,
                  this_time - frame.buffer_end_time,
                  FPS);
            LOG(INFO) << msg;
            last_pop_time = get_wall_time();
        }
        
        if (FLAGS_write_frames.empty()) {
            snprintf(tmp_str, 256, "%4.1f fps", FPS);
        } else {
            snprintf(tmp_str, 256, "%4.2f s/gpu", FLAGS_num_gpu*1.0/FPS);
        }
        
        // Print FPS and NumOfPeople on the images
        if (1) {
        cv::putText(wrap_frame, tmp_str, cv::Point(25,35),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255,150,150), 1);
        cv::putText(wrap_frame_right, tmp_str, cv::Point(25,35),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255,150,150), 1);

        snprintf(tmp_str, 256, "%4d", frame.numPeople);
        cv::putText(wrap_frame, tmp_str, cv::Point(DISPLAY_RESOLUTION_WIDTH-100+2, 35+2),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
        cv::putText(wrap_frame, tmp_str, cv::Point(DISPLAY_RESOLUTION_WIDTH-100, 35),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
        snprintf(tmp_str, 256, "%4d", frame_right.numPeople);
        cv::putText(wrap_frame_right, tmp_str, cv::Point(DISPLAY_RESOLUTION_WIDTH-100+2, 35+2),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0), 2);
        cv::putText(wrap_frame_right, tmp_str, cv::Point(DISPLAY_RESOLUTION_WIDTH-100, 35),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(150,150,255), 2);
        }

        if (global.part_to_show!=0) {
            if (global.part_to_show-1<=net_copies.at(0).up_model_descriptor->get_number_parts()) {
                snprintf(tmp_str, 256, "%10s", net_copies.at(0).up_model_descriptor->get_part_name(global.part_to_show-1).c_str());
            } else {
                int aff_part = ((global.part_to_show-1)-net_copies.at(0).up_model_descriptor->get_number_parts()-1)*2;
                if (aff_part==0) {
                    snprintf(tmp_str, 256, "%10s", "PAFs");
                } else {
                    aff_part = aff_part-2;
                    aff_part += 1+net_copies.at(0).up_model_descriptor->get_number_parts();
                    std::string uvname = net_copies.at(0).up_model_descriptor->get_part_name(aff_part);
                    std::string conn = uvname.substr(0, uvname.find("("));
                    snprintf(tmp_str, 256, "%10s", conn.c_str());
                }
            }
            cv::putText(wrap_frame, tmp_str, cv::Point(DISPLAY_RESOLUTION_WIDTH-175+1, 55+1),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
            cv::putText(wrap_frame_right, tmp_str, cv::Point(DISPLAY_RESOLUTION_WIDTH-175+1, 55+1),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
        }
        if (!FLAGS_video.empty() && FLAGS_write_frames.empty()) {
            snprintf(tmp_str, 256, "Frame %6d", global.uistate.current_frame);
            // cv::putText(wrap_frame, tmp_str, cv::Point(27,37),
            //     cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 2);
            cv::putText(wrap_frame, tmp_str, cv::Point(25,55),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255,255,255), 1);
            cv::putText(wrap_frame_right, tmp_str, cv::Point(25,55),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255,255,255), 1);
        }

        if (!FLAGS_no_display) {
            // cv::imshow("video", wrap_frame);
            publishStereoImgToROS(wrap_frame, wrap_frame_right);
            ROS_INFO("Published one pair of images.");
        }
        

        // Now publish the position of each joint.
        if (1) {
            	// printf("in the loop. \n");
            double scale = 1.0/frame.scale;
            const int num_parts = net_copies.at(0).up_model_descriptor->get_number_parts();
            char fname[256];
            // if (FLAGS_image_dir.empty()) {
                // sprintf(fname, "%s/frame%06d.json", FLAGS_write_json.c_str(), frame.video_frame_number);
            // } else {
                // boost::filesystem::path p(global.image_list[frame.video_frame_number]);
                // std::string rawname = p.stem().string();

                // sprintf(fname, "%s/%s.json", FLAGS_write_json.c_str(), rawname.c_str());
            // }
            // std::ofstream fs(fname);
            // fs << "{\n";
            // fs << "\"version\":0.1,\n";
            // fs << "\"bodies\":[\n";
            // for (int ip=0;ip<frame.numPeople;ip++) {
            //     fs << "{\n" << "\"joints\":" << "[";
            //     for (int ij=0;ij<num_parts;ij++) {
            //         fs << scale*frame.joints[ip*num_parts*3 + ij*3+0] << ",";
            //         fs << scale*frame.joints[ip*num_parts*3 + ij*3+1] << ",";
            //         fs << frame.joints[ip*num_parts*3 + ij*3+2];
            //         if (ij<num_parts-1) fs << ",";
            //     }
            //     fs << "]\n";
            //     fs << "}";
            //     if (ip<frame.numPeople-1) {
            //         fs<<",\n";
            //     }
            // }
            // fs << "]\n";
            // fs << "}\n";
            // // last_time += get_wall_time()-a;


    		//Publish the joint of each people in both image into ROS topic PUBLISH_DETECTION_NAME
            rtpose_ros::Detection detect_result;
            // Write left image result into detect_result
            for (int ip=0;ip<frame.numPeople;ip++) {
                for (int ij=0;ij<num_parts;ij++) {
                    //for loop, pushing data in the size of the array
                    detect_result.left_data.push_back(scale*frame.joints[ip*num_parts*3 + ij*3+0]);     // these three lines are used for publishing result into array, x pixel 
                    detect_result.left_data.push_back(scale*frame.joints[ip*num_parts*3 + ij*3+1]);     // these three lines are used for publishing result into array, y pixel 
                    detect_result.left_data.push_back(frame.joints[ip*num_parts*3 + ij*3+2]);           // these three lines are used for publishing result into array, confidence 
                }
            }
            detect_result.left_num = frame.numPeople;
            // Write right image result into detect_result
            for (int ip=0;ip<frame_right.numPeople;ip++) {
                for (int ij=0;ij<num_parts;ij++) {
                    //for loop, pushing data in the size of the array
                    detect_result.right_data.push_back(scale*frame_right.joints[ip*num_parts*3 + ij*3+0]);     // these three lines are used for publishing result into array, x pixel 
                    detect_result.right_data.push_back(scale*frame_right.joints[ip*num_parts*3 + ij*3+1]);     // these three lines are used for publishing result into array, y pixel 
                    detect_result.right_data.push_back(frame_right.joints[ip*num_parts*3 + ij*3+2]);           // these three lines are used for publishing result into array, confidence 
                }
            }
            detect_result.right_num = frame_right.numPeople;

            ros::Time now = ros::Time::now();
            detect_result.header.stamp = now;
            detect_result.header.frame_id = "zed_frame";
            poseDetectionPublisher.publish(detect_result);
        }


        display_counter++;

        delete [] frame.data_for_mat;
        delete [] frame.data_for_wrap;
        delete [] frame.data;
        delete [] frame_right.data_for_mat;
        delete [] frame_right.data_for_wrap;
        delete [] frame_right.data;

        //LOG(ERROR) << msg;
        int key = cv::waitKey(1);
        
        VLOG(2) << "Display time " << (get_wall_time()-tic)*1000.0 << " ms.";
    }
}

int rtcpm_stereo() {
    const auto timer_begin = std::chrono::high_resolution_clock::now();

    // // Opening processing deep net threads
    // pthread_t gpu_threads_pool[NUM_GPU];
    // for(int gpu = 0; gpu < NUM_GPU; gpu++) {
    //     int *arg = new int[1];
    //     *arg = gpu+FLAGS_start_device;
    //     int rc = pthread_create(&gpu_threads_pool[gpu], NULL, processFrame, (void *) arg);
    //     if (rc) {
    //         LOG(ERROR) << "Error:unable to create thread," << rc << "\n";
    //         return -1;
    //     }
    // }
    // LOG(INFO) << "Finish spawning " << NUM_GPU << " threads." << "\n";
    NUM_GPU = 2;
    // Opening processing deep net threads for left images
    pthread_t gpu_threads_pool[NUM_GPU];
    for(int gpu = 0; gpu < (NUM_GPU/2); gpu++) {
        int *arg = new int[1];
        *arg = gpu+FLAGS_start_device;
        int rc = pthread_create(&gpu_threads_pool[gpu], NULL, processLeftFrame, (void *) arg);
        if (rc) {
            LOG(ERROR) << "Error:unable to create thread for left images," << rc << "\n";
            return -1;
        }
        left_processing = false;
    }
    LOG(INFO) << "Finish spawning " << NUM_GPU << " threads." << "\n";
    // Opening processing deep net threads for right images
    // pthread_t gpu_threads_pool[NUM_GPU];
    for(int gpu = 1; gpu < NUM_GPU; gpu++) {
        int *arg = new int[1];
        *arg = gpu+FLAGS_start_device;
        int rc = pthread_create(&gpu_threads_pool[gpu], NULL, processRightFrame, (void *) arg);
        if (rc) {
            LOG(ERROR) << "Error:unable to create thread for right images," << rc << "\n";
            return -1;
        }
        right_processing = false;
    }
    LOG(INFO) << "Finish spawning " << NUM_GPU << " threads." << "\n";

    // // Setting output resolution
    // if (!FLAGS_no_display) {
    //     cv::namedWindow("video", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
    //     if (FLAGS_fullscreen) {
    //         cv::resizeWindow("video", 1920, 1080);
    //         cv::setWindowProperty("video", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    //         global.uistate.is_fullscreen = true;
    //     } else {
    //         cv::resizeWindow("video", DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT);
    //         cv::setWindowProperty("video", CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
    //         global.uistate.is_fullscreen = false;
    //     }
    // }

    // Openning frames producer (e.g. video, webcam) threads
    // usleep(3 * 1e6);
    // int thread_pool_size = 1;
    // pthread_t threads_pool[thread_pool_size];
    // for(int i = 0; i < thread_pool_size; i++) {
    //     int *arg = new int[i];
    //     int rc = pthread_create(&threads_pool[i], NULL, getFrameFromROS, (void *) arg);
    //     if (rc) {
    //         LOG(ERROR) << "Error: unable to create thread," << rc << "\n";
    //         return -1;
    //     }
    // }
    // VLOG(3) << "Finish spawning " << thread_pool_size << " threads. now waiting." << "\n";

    // // threads handling outputs
    // int thread_pool_size_out = NUM_GPU;
    // pthread_t threads_pool_out[thread_pool_size_out];
    // for(int i = 0; i < thread_pool_size_out; i++) {
    //     int *arg = new int[i];
    //     int rc = pthread_create(&threads_pool_out[i], NULL, postProcessFrame, (void *) arg);
    //     if (rc) {
    //         LOG(ERROR) << "Error: unable to create thread," << rc << "\n";
    //         return -1;
    //     }
    // }
    // VLOG(3) << "Finish spawning " << thread_pool_size_out << " threads. now waiting." << "\n";

    // threads handling outputs for left images
    int thread_pool_size_out = NUM_GPU-1;
    pthread_t threads_pool_out[thread_pool_size_out];
    for(int i = 0; i < thread_pool_size_out; i++) {
        int *arg = new int[i];
        int rc = pthread_create(&threads_pool_out[i], NULL, postProcessLeftFrame, (void *) arg);
        if (rc) {
            LOG(ERROR) << "Error: unable to create thread for left images," << rc << "\n";
            return -1;
        }
    }
    VLOG(3) << "Finish spawning " << thread_pool_size_out << " threads. now waiting." << "\n";

    // threads handling outputs for right images
    thread_pool_size_out = NUM_GPU;
    // pthread_t threads_pool_out[thread_pool_size_out];
    for(int i = 1; i < thread_pool_size_out; i++) {
        int *arg = new int[i];
        int rc = pthread_create(&threads_pool_out[i], NULL, postProcessRightFrame, (void *) arg);
        if (rc) {
            LOG(ERROR) << "Error: unable to create thread for right images," << rc << "\n";
            return -1;
        }
    }
    VLOG(3) << "Finish spawning " << thread_pool_size_out << " threads. now waiting." << "\n";



    // thread for buffer and ordering frame
    int buffer_num = 2;
    pthread_t threads_order[buffer_num];
    int *arg = new int[1];
    int rc = pthread_create(&threads_order[0], NULL, buffer_and_order_left, (void *) arg);
    if (rc) {
        LOG(ERROR) << "Error: unable to create thread for left buffer_and_order," << rc << "\n";
        return -1;
    }
    VLOG(3) << "Finish spawning the thread for left images ordering. now waiting." << "\n";

    // thread for buffer and ordering frame
    // pthread_t threads_order;
    arg = new int[1];
    rc = pthread_create(&threads_order[1], NULL, buffer_and_order_right, (void *) arg);
    if (rc) {
        LOG(ERROR) << "Error: unable to create thread for right buffer_and_order," << rc << "\n";
        return -1;
    }
    VLOG(3) << "Finish spawning the thread for right images ordering. now waiting." << "\n";

    // display
    // pthread_t thread_display;
    // rc = pthread_create(&thread_display, NULL, displayROSFrame, (void *) arg);
    // if (rc) {
    //     LOG(ERROR) << "Error: unable to create thread," << rc << "\n";
    //     return -1;
    // }
    // VLOG(3) << "Finish spawning the thread for display. now waiting." << "\n";

    // // Joining threads
    // for (int i = 0; i < thread_pool_size; i++) {
    //     pthread_join(threads_pool[i], NULL);
    // }
    // for (int i = 0; i < NUM_GPU; i++) {
    //     pthread_join(gpu_threads_pool[i], NULL);
    // }

    ROS_INFO("All rtcpm_stereo threads successfully created.");

    const auto total_time_sec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-timer_begin).count() * 1e-9;
    // LOG(ERROR) << "Total time: " << total_time_sec << " seconds.";
    ROS_INFO("Creating threads time: %f seconds.", total_time_sec);

    return 0;
}

bool handleKey(int c) {
    const std::string key2part = "0123456789qwertyuiopas";
    VLOG(4) << "key: " << (char)c << " code: " << c;
    if (c>=65505) {
        global.uistate.is_shift_down = true;
        c = (char)c;
        c = tolower(c);
        VLOG(4) << "post key: " << (char)c << " code: " << c;
    } else {
        global.uistate.is_shift_down = false;
    }
    VLOG(4) << "shift: " << global.uistate.is_shift_down;
    if (c==27) {
        global.quit_threads = true;
        return false;
    }

    if (c=='g') {
        global.uistate.is_googly_eyes = !global.uistate.is_googly_eyes;
    }

    // Rudimentary seeking in video
    if (c=='l' || c=='k' || c==' ') {
        if (!FLAGS_video.empty()) {
            int cur_frame = global.uistate.current_frame;
            int frame_delta = 30;
            if (global.uistate.is_shift_down)
                frame_delta = 2;
            if (c=='l') {
                VLOG(4) << "Jump " << frame_delta << " frames to " << cur_frame;
                global.uistate.current_frame+=frame_delta;
                global.uistate.seek_to_frame = 1;
            } else if (c=='k') {
                VLOG(4) << "Rewind " << frame_delta << " frames to " << cur_frame;
                global.uistate.current_frame-=frame_delta;
                global.uistate.seek_to_frame = 1;
            }
        }
        if (c==' ') {
            global.uistate.is_video_paused = !global.uistate.is_video_paused;
        }
    }
    if (c=='f') {
        if (!global.uistate.is_fullscreen) {
            cv::namedWindow("video", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
            cv::resizeWindow("video", 1920, 1080);
            cv::setWindowProperty("video", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            global.uistate.is_fullscreen = true;
        } else {
            cv::namedWindow("video", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
            cv::setWindowProperty("video", CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
            cv::resizeWindow("video", DISPLAY_RESOLUTION_WIDTH, DISPLAY_RESOLUTION_HEIGHT);
            global.uistate.is_fullscreen = false;
        }
    }
    int target = -1;
    int ind = key2part.find(c);
    if (ind!=std::string::npos) {// && !global.uistate.is_shift_down) {
        target = ind;
    }

    if (target >= 0 && target <= 42) {
        global.part_to_show = target;
        LOG(INFO) << "p2s: " << global.part_to_show;
    }

    if (c=='-' || c=='=') {
        if (c=='-')
            global.nms_threshold -= 0.005;
        if (c=='=')
            global.nms_threshold += 0.005;
        LOG(INFO) << "nms_threshold: " << global.nms_threshold;
    }
    if (c=='_' || c=='+') {
        if (c=='_')
            global.connect_min_subset_score -= 0.005;
        if (c=='+')
            global.connect_min_subset_score += 0.005;
        LOG(INFO) << "connect_min_subset_score: " << global.connect_min_subset_score;
    }
    if (c=='[' || c==']') {
        if (c=='[')
            global.connect_inter_threshold -= 0.005;
        if (c==']')
            global.connect_inter_threshold += 0.005;
        LOG(INFO) << "connect_inter_threshold: " << global.connect_inter_threshold;
    }
    if (c=='{' || c=='}') {
        if (c=='{')
            global.connect_inter_min_above_threshold -= 1;
        if (c=='}')
            global.connect_inter_min_above_threshold += 1;
        LOG(INFO) << "connect_inter_min_above_threshold: " << global.connect_inter_min_above_threshold;
    }
    if (c==';' || c=='\'') {
        if (c==';')
        global.connect_min_subset_cnt -= 1;
        if (c=='\'')
            global.connect_min_subset_cnt += 1;
        LOG(INFO) << "connect_min_subset_cnt: " << global.connect_min_subset_cnt;
    }

    if (c==',' || c=='.') {
        if (c=='.')
            global.part_to_show++;
        if (c==',')
            global.part_to_show--;
        if (global.part_to_show<0) {
            global.part_to_show = 42;
        }
        // if (global.part_to_show>42) {
        //     global.part_to_show = 0;
        // }
        if (global.part_to_show>55) {
            global.part_to_show = 0;
        }
        LOG(INFO) << "p2s: " << global.part_to_show;
    }

    return true;
}

// The global parameters must be assign after the main has started, not statically before. Otherwise, they will take the default values, not the user-introduced values.
int setGlobalParametersFromFlags() {
    int nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d", &DISPLAY_RESOLUTION_WIDTH, &DISPLAY_RESOLUTION_HEIGHT);
    CHECK_EQ(nRead,2) << "Error, resolution format (" <<  FLAGS_resolution << ") invalid, should be e.g., 960x540 ";
    if (DISPLAY_RESOLUTION_WIDTH==-1 && !FLAGS_video.empty()) {
        cv::VideoCapture cap;
        CHECK(cap.open(FLAGS_video)) << "Couldn't open video " << FLAGS_video;
        DISPLAY_RESOLUTION_WIDTH = cap.get(CV_CAP_PROP_FRAME_WIDTH);
        DISPLAY_RESOLUTION_HEIGHT = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        LOG(INFO) << "Setting display resolution from video: " << DISPLAY_RESOLUTION_WIDTH << "x" << DISPLAY_RESOLUTION_HEIGHT;
    } else if (DISPLAY_RESOLUTION_WIDTH==-1 && !FLAGS_image_dir.empty()) {
        cv::Mat image_uchar_orig = cv::imread(global.image_list[0].c_str(), CV_LOAD_IMAGE_COLOR);
        DISPLAY_RESOLUTION_WIDTH = image_uchar_orig.cols;
        DISPLAY_RESOLUTION_HEIGHT = image_uchar_orig.rows;
        LOG(INFO) << "Setting display resolution from first image: " << DISPLAY_RESOLUTION_WIDTH << "x" << DISPLAY_RESOLUTION_HEIGHT;
    } else if (DISPLAY_RESOLUTION_WIDTH==-1) {
        LOG(ERROR) << "Invalid resolution without video/images: " << DISPLAY_RESOLUTION_WIDTH << "x" << DISPLAY_RESOLUTION_HEIGHT;
        exit(1);
    } else {
        LOG(INFO) << "Display resolution: " << DISPLAY_RESOLUTION_WIDTH << "x" << DISPLAY_RESOLUTION_HEIGHT;
    }
    nRead = sscanf(FLAGS_camera_resolution.c_str(), "%dx%d", &CAMERA_FRAME_WIDTH, &CAMERA_FRAME_HEIGHT);
    CHECK_EQ(nRead,2) << "Error, camera resolution format (" <<  FLAGS_camera_resolution << ") invalid, should be e.g., 1280x720";
    nRead = sscanf(FLAGS_net_resolution.c_str(), "%dx%d", &NET_RESOLUTION_WIDTH, &NET_RESOLUTION_HEIGHT);
    CHECK_EQ(nRead,2) << "Error, net resolution format (" <<  FLAGS_net_resolution << ") invalid, should be e.g., 656x368 (multiples of 16)";
    LOG(INFO) << "Net resolution: " << NET_RESOLUTION_WIDTH << "x" << NET_RESOLUTION_HEIGHT;

    if (!FLAGS_write_frames.empty()) {
        // Create folder if it does not exist
        boost::filesystem::path dir(FLAGS_write_frames);
        if (!boost::filesystem::is_directory(dir) && !boost::filesystem::create_directory(dir)) {
            LOG(ERROR) << "Could not write to or create directory " << dir;
            return 1;
        }
    }

    if (!FLAGS_write_json.empty()) {
        // Create folder if it does not exist
        boost::filesystem::path dir(FLAGS_write_json);
        if (!boost::filesystem::is_directory(dir) && !boost::filesystem::create_directory(dir)) {
            LOG(ERROR) << "Could not write to or create directory " << dir;
            return 1;
        }
    }

    BATCH_SIZE = {FLAGS_num_scales};
    SCALE_GAP = {FLAGS_scale_gap};
    START_SCALE = {FLAGS_start_scale};
    NUM_GPU = {FLAGS_num_gpu};
    // Global struct/classes
    global.part_to_show = FLAGS_part_to_show;
    net_copies = std::vector<NetCopy>(NUM_GPU);
    // Set nets
    PERSON_DETECTOR_CAFFEMODEL = FLAGS_caffemodel;
    PERSON_DETECTOR_PROTO = FLAGS_caffeproto;

    return 0;
}

int readImageDirIfFlagEnabled()
{
    // Open & read image dir if present
    if (!FLAGS_image_dir.empty()) {
        std::string folderName = FLAGS_image_dir;
        if ( !boost::filesystem::exists( folderName ) ) {
            LOG(ERROR) << "Folder " << folderName << " does not exist.";
            return -1;
        }
        boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
        for ( boost::filesystem::directory_iterator itr( folderName ); itr != end_itr; ++itr ) {
            if ( boost::filesystem::is_directory(itr->status()) ) {
                // Skip directories
            } else if (itr->path().extension()==".jpg" || itr->path().extension()==".png" || itr->path().extension()==".bmp") {
                //  std::string filename = itr->path().string();
                global.image_list.push_back( itr->path().string() );
            }
        }
        std::sort(global.image_list.begin(), global.image_list.end());
        CHECK_GE(global.image_list.size(),0);
    }

    return 0;
}

void compute_epilines(cv::Mat points, int img_case, cv::Matx<float,3,3>  Fund_matrix, std::vector<cv::Vec<float,3>> &epilines){
    // std::cout << "points size is: "<< points.size() << std::endl;
    // std::cout << "points.rows is: "<< points.rows << std::endl;
    if (img_case==1){
        cv::transpose(Fund_matrix, Fund_matrix);
    }
    // std::cout << "Transposed Fund_matrix is: "<< Fund_matrix << std::endl;

    float* mp = &points.at<float>(0);
    float x, y;
    cv::Vec<float,3> now_vec;
    for (int row=0; row<points.rows; row++) {
        x = mp[row*2];
        y = mp[row*2 + 1];
        for (int i=0; i<3; i++){
            now_vec[i] = x*Fund_matrix(0,i) + y*Fund_matrix(1,i) + Fund_matrix(2,i);
        }
        epilines.push_back(now_vec);
        // std::cout << "epilines.size() is: "<< epilines.size() << std::endl;
        // std::cout << "epilines["<< row <<"] is: "<< epilines[row] << std::endl;
    }
    // std::cout << "epilines is: "<< epilines << std::endl;
}

void reconstruct_3d_pose(const rtpose_ros::Detection detect_result)  {
    ROS_INFO("Left image has %i people.",detect_result.left_num);
    ROS_INFO("Right image has %i people.",detect_result.right_num);
    int left_num = detect_result.left_num;
    int right_num = detect_result.right_num;
    const int num_parts = net_copies.at(0).up_model_descriptor->get_number_parts();

        

    if(detect_result.left_num > 0 && detect_result.right_num > 0){          //Make sure there are people on both images


        cv::Matx<float,3,3> Fund_matrix;
        // cv::Mat Fund_matrix = cv::Mat(3,3,CV_64F);
        Fund_matrix(0,0) = -9.09430389e-10;
        Fund_matrix(0,1) = 2.70030811e-06;
        Fund_matrix(0,2) = -9.16306854e-04;
        Fund_matrix(1,0) = -1.82561938e-06;
        Fund_matrix(1,1) = 3.75124867e-07;
        Fund_matrix(1,2) = 0.18062506;
        Fund_matrix(2,0) = 6.80660378e-04;
        Fund_matrix(2,1) = -0.18121823;
        Fund_matrix(2,2) = 0.14064993;
        
        std::cout << "Fund_matrix is: "<< Fund_matrix << std::endl;

        // Camera1 is the left camera, camera2 is thre right camera
        std::vector<cv::Vec<float,3>> epilines1, epilines2;
        // cv::Mat points1 = cv::Mat(left_num*num_parts, 2, CV_32F);     //like usual matrix, cv::Mat put rows first. such as cv::Mat(rows, cols, data_format),  row major order
        // cv::Mat points2 = cv::Mat(right_num*num_parts, 2, CV_32F);    //like usual matrix, cv::Mat put rows first. such as cv::Mat(rows, cols, data_format)
        // float* mp1 = &points1.at<float>(0);
        // float* mp2 = &points2.at<float>(0);
        // // // Convert left image result to Nx2 matrix: points1
        // for (int ip=0; ip<left_num; ip++) {
        //     for (int ij=0;ij<num_parts;ij++) {
        //         float x_pixel = float(detect_result.left_data[ip*num_parts*3 + ij*3+0]);
        //         float y_pixel = float(detect_result.left_data[ip*num_parts*3 + ij*3+1]);
        //         std::cout << "detect_result.left_data: "<< x_pixel << "," << y_pixel << std::endl;
        //         // std::cout << "detect_result.left_data, float: "<< float(detect_result.left_data[ip*num_parts*3 + ij*3+0]) << std::endl;
        //         // point1 is a Nx2 marix and data are stored in a row major order, which means points1.data[1] is the second number in the first row!!
        //         mp1[(ip*num_parts+ij)*2] = float(x_pixel);          // x pixel 
        //         mp1[(ip*num_parts+ij)*2 + 1] = float(y_pixel);      // y pixel 
        //     }
        // }
        // // // Convert right image result to Nx2 matrix: points2
        // for (int ip=0; ip<right_num; ip++) {
        //     for (int ij=0;ij<num_parts;ij++) {
        //         float x_pixel = detect_result.right_data[ip*num_parts*3 + ij*3+0];
        //         float y_pixel = detect_result.right_data[ip*num_parts*3 + ij*3+1];
        //         std::cout << "detect_result.right_data: "<< x_pixel << "," << y_pixel << std::endl;
        //         mp2[(ip*num_parts+ij)*2] = float(x_pixel);         // x pixel 
        //         mp2[(ip*num_parts+ij)*2 +1] = float(y_pixel);      // y pixel  
        //     }
        // }

        // std::cout << "Fund_matrix is: "<< Fund_matrix << std::endl;
        // std::cout << "points1.size() is: "<< points1.size() << std::endl;
        // std::cout << "sizeof(CV_32F) is: "<< sizeof(CV_32F) << std::endl;

        ///////////////////// For debugging ///////////////////////////////////
        // x_left_now =

        //    648   263
        //    651   310
        //    600   310
        //    536   292
        //    456   264
        //    701   309
        //    769   292
        //    838   271
        //    633   454
        //    649   568
        //    658   675
        //    689   441
        //    697   483
        //    702   579
        //    638   255
        //    658   255
        //    623   260
        //    671   259

        // epiLines =

        //    -0.0002    0.1795  -47.0787
        //    -0.0001    0.1796  -55.5939
        //    -0.0001    0.1796  -55.6286
        //    -0.0001    0.1798  -52.4102
        //    -0.0002    0.1799  -47.3906
        //    -0.0001    0.1795  -55.3786
        //    -0.0001    0.1793  -52.2516
        //    -0.0002    0.1792  -48.3991
        //     0.0003    0.1796  -81.7016
        //     0.0006    0.1797 -102.3496
        //     0.0009    0.1797 -121.7338
        //     0.0003    0.1795  -79.3076
        //     0.0004    0.1795  -86.9133
        //     0.0006    0.1796 -104.3069
        //    -0.0002    0.1796  -45.6357
        //    -0.0002    0.1795  -45.6221
        //    -0.0002    0.1796  -46.5520
        //    -0.0002    0.1795  -46.3382

           // x_right_1 =

           // 610   264
           // 615   310
           // 565   311
           // 500   293
           // 418   264
           // 665   309
           // 734   293
           // 801   271
           // 598   453
           // 613   569
           // 623   676
           // 652   440
           // 654   484
           // 663   575
           // 600   255
           // 620   255
           // 589   261
           // 636   260

        left_num = 1;
        right_num = 1;

        cv::Mat points1 = cv::Mat(18, 2, CV_32F);
        float* mp1 = &points1.at<float>(0);
        for (int ip=0; ip<1; ip++) {
            mp1[0] = float(648);       // x pixel 
            mp1[1] = float(263);      // y pixel 
            mp1[2] = float(651);       // x pixel 
            mp1[3] = float(310);      // y pixel 
            mp1[4] = float(600);       // x pixel 
            mp1[5] = float(310);      // y pixel 
            mp1[6] = float(536);       // x pixel 
            mp1[7] = float(292);      // y pixel 
            mp1[8] = float(456);       // x pixel 
            mp1[9] = float(264);      // y pixel 
            mp1[10] = float(701);       // x pixel 
            mp1[11] = float(309);      // y pixel 
            mp1[12] = float(769);       // x pixel 
            mp1[13] = float(292);      // y pixel 
            mp1[14] = float(838);       // x pixel 
            mp1[15] = float(271);      // y pixel 
            mp1[16] = float(633);       // x pixel 
            mp1[17] = float(454);      // y pixel 
            mp1[18] = float(649);       // x pixel 
            mp1[19] = float(568);      // y pixel 
            mp1[20] = float(658);       // x pixel 
            mp1[21] = float(675);      // y pixel 
            mp1[22] = float(689);       // x pixel 
            mp1[23] = float(441);      // y pixel 
            mp1[24] = float(697);       // x pixel 
            mp1[25] = float(483);      // y pixel 
            mp1[26] = float(702);       // x pixel 
            mp1[27] = float(579);      // y pixel 
            mp1[28] = float(638);       // x pixel 
            mp1[29] = float(255);      // y pixel 
            mp1[30] = float(658);       // x pixel 
            mp1[31] = float(255);      // y pixel 
            mp1[32] = float(623);       // x pixel 
            mp1[33] = float(260);      // y pixel 
            mp1[34] = float(671);       // x pixel 
            mp1[35] = float(259);      // y pixel 
        }
        cv::Mat points2 = cv::Mat(18, 2, CV_32F);
        float* mp2 = &points2.at<float>(0);
        for (int ip=0; ip<1; ip++) {
            mp2[0] = float(610);       // x pixel 
            mp2[1] = float(264);      // y pixel 
            mp2[2] = float(615);       // x pixel 
            mp2[3] = float(310);      // y pixel 
            mp2[4] = float(565);       // x pixel 
            mp2[5] = float(311);      // y pixel 
            mp2[6] = float(500);       // x pixel 
            mp2[7] = float(293);      // y pixel 
            mp2[8] = float(418);       // x pixel 
            mp2[9] = float(264);      // y pixel 
            mp2[10] = float(665);       // x pixel 
            mp2[11] = float(309);      // y pixel 
            mp2[12] = float(734);       // x pixel 
            mp2[13] = float(293);      // y pixel 
            mp2[14] = float(801);       // x pixel 
            mp2[15] = float(271);      // y pixel 
            mp2[16] = float(598);       // x pixel 
            mp2[17] = float(454);      // y pixel 
            mp2[18] = float(613);       // x pixel 
            mp2[19] = float(569);      // y pixel 
            mp2[20] = float(623);       // x pixel 
            mp2[21] = float(676);      // y pixel 
            mp2[22] = float(652);       // x pixel 
            mp2[23] = float(440);      // y pixel 
            mp2[24] = float(654);       // x pixel 
            mp2[25] = float(484);      // y pixel 
            mp2[26] = float(663);       // x pixel 
            mp2[27] = float(575);      // y pixel 
            mp2[28] = float(600);       // x pixel 
            mp2[29] = float(255);      // y pixel 
            mp2[30] = float(620);       // x pixel 
            mp2[31] = float(255);      // y pixel 
            mp2[32] = float(589);       // x pixel 
            mp2[33] = float(261);      // y pixel 
            mp2[34] = float(636);       // x pixel 
            mp2[35] = float(260);      // y pixel 
        }
        ///////////////////// For debugging ///////////////////////////////////
        compute_epilines(points1, 1, Fund_matrix, epilines1);  // input is pts on the left image, output is the correspond epilines on the right image
        // std::cout << "epilines1.size() is: "<< epilines1.size() << std::endl;
        vector< vector<double> > costMatrix;
        double infinity_cost = 1000;
        for (int il=0; il<left_num; il++) {
            vector<double> cost_row;
            for (int ir=0; ir<right_num; ir++) {
                float cost = 0;
                int useful_num = 0;
                for(int k=0; k<num_parts; k++){          //goes through all cv::Vec<float,3> in the std::vector
                    // std::cout << "points1 is: "<< points1.at<float>(2*k) << "," << points1.at<float>(2*k+1) << std::endl;
                    // std::cout << "points2 is: "<< points2.at<float>(2*k) << "," << points2.at<float>(2*k+1) << std::endl;
                    if (points1.at<float>(2*(il*num_parts+k))>0 && points1.at<float>(2*(il*num_parts+k)+1)>0 
                        && points2.at<float>(2*(ir*num_parts+k))>0 && points2.at<float>(2*(ir*num_parts+k)+1)>0){
                        std::cout << "points1 is: "<< points1.at<float>(2*(il*num_parts+k)) << "," << points1.at<float>(2*(il*num_parts+k)+1) << std::endl;
                        std::cout << "epilines1["<< il*num_parts+k <<"] is: "<< epilines1[il*num_parts+k] << std::endl;
                        std::cout << "points2 is: "<< points2.at<float>(2*(ir*num_parts+k)) << "," << points2.at<float>(2*(ir*num_parts+k)+1) << std::endl;
                        float now_cost = abs(points2.at<float>(2*(ir*num_parts+k))*epilines1[il*num_parts+k][0] + points2.at<float>(2*(ir*num_parts+k)+1)*epilines1[il*num_parts+k][1] + epilines1[il*num_parts+k][2]);
                        std::cout << "for epiline ["<< il*num_parts+k <<"], cost is: "<< now_cost << std::endl;
                        cost = cost + now_cost;
                        useful_num++;
                    }
                }
                if(useful_num != 0){
                    cost_row.push_back(double(cost/useful_num));
                }
                else{
                    cost_row.push_back(infinity_cost);
                }
                // Since Hungarian need the cost matrix to be a square matrix, we should fill the blank with infinity_cost if rows are more than columns
                if(left_num>right_num){
                    for(int i=0; i<(left_num-right_num); i++){
                        cost_row.push_back(infinity_cost);
                    }
                }
            }
            costMatrix.push_back(cost_row);
        }
        // Since Hungarian need the cost matrix to be a square matrix, we should fill the blank with infinity_cost if columns are more than rows
        if(left_num<right_num){
            for(int j=0; j<(right_num-left_num); j++){
                vector<double> cost_row;
                for(int i=0; i<right_num; i++){
                    cost_row.push_back(infinity_cost);
                }
                costMatrix.push_back(cost_row);
            }
        }
        std::cout << "costMatrix row size() is: " << costMatrix.size() << std::endl;
        std::cout << "costMatrix col size() is: " << costMatrix[0].size() << std::endl;
        // std::cout << "costMatrix is: " << costMatrix[0][0] << std::endl;
        std::cout << "costMatrix is: "<< std::endl;
        std::cout << "[";
        for (unsigned int x = 0; x < costMatrix.size(); x++){
            for (unsigned int y = 0; y < costMatrix[0].size(); y++){
                std::cout << costMatrix[x][y];
                if(y == costMatrix[0].size()-1){
                    std::cout << ";";
                    if(x < costMatrix.size()-1){
                        std::cout << std::endl;
                    } 
                }
                else{
                    std::cout << ",";
                }
            }
        }
        std::cout << "]" << std::endl;

        // Data association
        HungarianAlgorithm HungAlgo;
        vector<int> assignment;
        double cost_total = HungAlgo.Solve(costMatrix, assignment);
        std::cout << "cost_total is: " << cost_total << std::endl;

        // Now, rearrange the detection results to two new Mat variables "points1_new" and "points2_new", and only keep the associated data
        double cost_of_non_assignment = 2;
        int useful_pair = 0;
        vector<int> useful_left_index;
        for (unsigned int x = 0; x <left_num; x++){
            if(x<left_num && assignment[x]<right_num && costMatrix[x][assignment[x]]<cost_of_non_assignment){
                useful_pair++;
                useful_left_index.push_back(x);
                std::cout << "assignment : " << x << "," << assignment[x] << std::endl;
            }
            
        }
        if(useful_pair == 0){
            // consider a situation that all elements in the cost_matrix are infinity, which means no useful pairing result.
            // In this case, we should just jump out of this function.
            return;
        }

        // For cv::triangulatePoints, rows have to be == 2 of the input points.
        cv::Mat points1_new = cv::Mat(2, useful_pair*num_parts, CV_32F);
        float* mp1_new = &points1_new.at<float>(0);
        cv::Mat points2_new = cv::Mat(2, useful_pair*num_parts, CV_32F);
        float* mp2_new = &points2_new.at<float>(0);
        for (int i = 0; i < useful_pair; i++){
            for (int ip=0; ip<num_parts; ip++) {
                mp1_new[i*num_parts + ip] = mp1[useful_left_index[i]*num_parts + 2*ip];
                mp1_new[useful_pair*num_parts + i*num_parts + ip] = mp1[useful_left_index[i]*num_parts + 2*ip + 1];
                mp2_new[i*num_parts + ip] = mp2[assignment[i]*num_parts + 2*ip];
                mp2_new[useful_pair*num_parts + i*num_parts + ip] = mp2[assignment[i]*num_parts + 2*ip + 1];
                std::cout << "mp1_new[" << i*num_parts + ip<< "] is :"<< mp1_new[i*num_parts + ip]  << std::endl;
                std::cout << "mp1_new[" << useful_pair*num_parts + i*num_parts + ip<< "] is :"<< mp1_new[useful_pair*num_parts + i*num_parts + ip]  << std::endl;
            }
        }

        cv::Mat_<double> Camera_P1 = (cv::Mat_<double>(3,4) <<
                        669.0622,   0,              677.9941,       0,
                        0,          669.0140,       361.2724,       0,
                        0,          0,              1,              0);
        cv::Mat_<double> Camera_P2 = (cv::Mat_<double>(3,4) <<
                        670.1632298,1.2470508,      6.762432090424326e+02,      -8.065791024010150e+04,
                        0.8807893,  668.52436137,   3.611070472550634e+02,      -3.046998430787702e+02,
                        0.00325779, 0.0014024390,   1,                          -0.815933154416481);
        // cout << "P1\n" << cv::Mat(Camera_P0) << endl;
        cout << "Camera_P0\n" << Camera_P1 << endl;
        cout << "points1_new\n" << points1_new << endl;
        cout << "points2_new\n" << points2_new << endl;
        cv::Mat pt_3d_h(4,useful_pair,CV_32FC1);

        cv::triangulatePoints(Camera_P1, Camera_P2, points1_new, points2_new, pt_3d_h);
        // Since the triangulation uses a SVD to compute the solution, the points in pt_3d_h are normalized to unit vectors.
        std::cout << "pt_3d_h is: " << pt_3d_h << std::endl;
        // std::cout << "pt_3d_h.at<float>(0) is: " << pt_3d_h.at<float>(1,15) << std::endl;
        // vector<cv::Point3f> pt_3d;
        // cv::convertPointsFromHomogeneous(pt_3d_h.reshape(4, 1), pt_3d);
        // std::cout << "pt_3d is: " << pt_3d << std::endl;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_3d(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (int i=0; i<useful_pair; i++) {
            // push back each joint
            for (int j=0; j<num_parts; j++){
                if(mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0){
                    pcl::PointXYZRGB point;
                    point.r = 255;
                    point.g = 255;
                    point.b = 255;
                    point.y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000; //Unmornalize and then convert from mm to meter
                    point.z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000; //Unmornalize and then convert from mm to meter
                    point.x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000; //Unmornalize and then convert from mm to meter
                    std::cout << "point is: " << point << std::endl;
                    cloud_3d->points.push_back (point);
                }
            }
            // create pointclouds for limbs if it is detected
            for(int j=0; j<num_parts; j++){
                pcl::PointXYZRGB point;
                float start_x, start_y, start_z;
                float end_x, end_y, end_z;
                int pt_in_line = 10;
                // Connect Nose to Neck
                if(j==0 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 1]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 1]>0 && 
                    mp2_new[i*num_parts + j + 1]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 1]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    point.r = 255*1.000;
                    point.g = 255*0;
                    point.b = 255*0;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect Neck to RShoulder
                if(j==1 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 1]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 1]>0 && 
                    mp2_new[i*num_parts + j + 1]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 1]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    point.r = 255*1.000;
                    point.g = 255*0.3158;
                    point.b = 255*0;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect RShoulder to RElbow
                if(j==2 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 1]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 1]>0 && 
                    mp2_new[i*num_parts + j + 1]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 1]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    point.r = 255*1.000;
                    point.g = 255*0.6316;
                    point.b = 255*0;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect RElbow to RWrist
                if(j==3 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 1]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 1]>0 && 
                    mp2_new[i*num_parts + j + 1]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 1]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    point.r = 255*1.000;
                    point.g = 255*0.9474;
                    point.b = 255*0;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect Neck to LShoulder
                if(j==1 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 4]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 4]>0 && 
                    mp2_new[i*num_parts + j + 4]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 4]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+4)/pt_3d_h.at<float>(3,i*num_parts + j+4)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+4)/pt_3d_h.at<float>(3,i*num_parts + j+4)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+4)/pt_3d_h.at<float>(3,i*num_parts + j+4)/1000;
                    point.r = 255*0.7368;
                    point.g = 255*1.000;
                    point.b = 255*0;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect LShoulder to LElbow
                if(j==5 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 1]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 1]>0 && 
                    mp2_new[i*num_parts + j + 1]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 1]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    point.r = 255*0.4211;
                    point.g = 255*1.000;
                    point.b = 255*0;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect LElbow to LWrist
                if(j==6 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 1]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 1]>0 && 
                    mp2_new[i*num_parts + j + 1]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 1]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    point.r = 255*0.1053;
                    point.g = 255*1.000;
                    point.b = 255*0;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect Neck to RHip
                if(j==1 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 7]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 7]>0 && 
                    mp2_new[i*num_parts + j + 7]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 7]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+7)/pt_3d_h.at<float>(3,i*num_parts + j+7)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+7)/pt_3d_h.at<float>(3,i*num_parts + j+7)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+7)/pt_3d_h.at<float>(3,i*num_parts + j+7)/1000;
                    point.r = 255*0;
                    point.g = 255*1.000;
                    point.b = 255*0.2105;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect RHip to RKnee
                if(j==8 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 1]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 1]>0 && 
                    mp2_new[i*num_parts + j + 1]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 1]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    point.r = 255*0;
                    point.g = 255*1.000;
                    point.b = 255*0.5263;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect RKnee to RAnkle
                if(j==9 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 1]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 1]>0 && 
                    mp2_new[i*num_parts + j + 1]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 1]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    point.r = 255*0;
                    point.g = 255*1.000;
                    point.b = 255*0.8421;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect Neck to LHip
                if(j==1 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 10]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 10]>0 && 
                    mp2_new[i*num_parts + j + 10]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 10]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+10)/pt_3d_h.at<float>(3,i*num_parts + j+10)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+10)/pt_3d_h.at<float>(3,i*num_parts + j+10)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+10)/pt_3d_h.at<float>(3,i*num_parts + j+10)/1000;
                    point.r = 255*0;
                    point.g = 255*0.8421;
                    point.b = 255*1.0000;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect LHip to LKnee
                if(j==11 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 1]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 1]>0 && 
                    mp2_new[i*num_parts + j + 1]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 1]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    point.r = 255*0;
                    point.g = 255*0.5263;
                    point.b = 255*1.0000;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect LKnee to LAnkle
                if(j==12 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 1]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 1]>0 && 
                    mp2_new[i*num_parts + j + 1]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 1]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+1)/pt_3d_h.at<float>(3,i*num_parts + j+1)/1000;
                    point.r = 255*0;
                    point.g = 255*0.2105;
                    point.b = 255*1.0000;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect Nose to REye
                if(j==0 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j +14]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j +14]>0 && 
                    mp2_new[i*num_parts + j +14]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j +14]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+14)/pt_3d_h.at<float>(3,i*num_parts + j+14)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+14)/pt_3d_h.at<float>(3,i*num_parts + j+14)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+14)/pt_3d_h.at<float>(3,i*num_parts + j+14)/1000;
                    point.r = 255*0.1053;
                    point.g = 255*0;
                    point.b = 255*1.0000;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect Nose to LEye
                if(j==0 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j + 15]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j + 15]>0 && 
                    mp2_new[i*num_parts + j + 15]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j + 15]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+15)/pt_3d_h.at<float>(3,i*num_parts + j+15)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+15)/pt_3d_h.at<float>(3,i*num_parts + j+15)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+15)/pt_3d_h.at<float>(3,i*num_parts + j+15)/1000;
                    point.r = 255*0.4211;
                    point.g = 255*0;
                    point.b = 255*1.0000;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect REye to REar
                if(j==14 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j +2]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j +2]>0 && 
                    mp2_new[i*num_parts + j +2]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j +2]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+2)/pt_3d_h.at<float>(3,i*num_parts + j+2)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+2)/pt_3d_h.at<float>(3,i*num_parts + j+2)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+2)/pt_3d_h.at<float>(3,i*num_parts + j+2)/1000;
                    point.r = 255*0.7368;
                    point.g = 255*0;
                    point.b = 255*1.0000;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
                // Connect LEye to LEar
                if(j==15 && mp1_new[i*num_parts + j]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j]>0 && 
                    mp2_new[i*num_parts + j]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j]>0 &&
                    mp1_new[i*num_parts + j +2]>0 && mp1_new[useful_pair*num_parts + i*num_parts + j +2]>0 && 
                    mp2_new[i*num_parts + j +2]>0 && mp2_new[useful_pair*num_parts + i*num_parts + j +2]>0){
                    start_x = pt_3d_h.at<float>(2,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_y = -pt_3d_h.at<float>(0,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    start_z = -pt_3d_h.at<float>(1,i*num_parts + j)/pt_3d_h.at<float>(3,i*num_parts + j)/1000;
                    end_x = pt_3d_h.at<float>(2,i*num_parts + j+2)/pt_3d_h.at<float>(3,i*num_parts + j+2)/1000;
                    end_y = -pt_3d_h.at<float>(0,i*num_parts + j+2)/pt_3d_h.at<float>(3,i*num_parts + j+2)/1000;
                    end_z = -pt_3d_h.at<float>(1,i*num_parts + j+2)/pt_3d_h.at<float>(3,i*num_parts + j+2)/1000;
                    point.r = 255*1.0000;
                    point.g = 255*0;
                    point.b = 255*0.9474;
                    for(int k=1; k<pt_in_line; k++){
                        point.x = ((end_x-start_x)/pt_in_line)*k+start_x;
                        point.y = ((end_y-start_y)/pt_in_line)*k+start_y;
                        point.z = ((end_z-start_z)/pt_in_line)*k+start_z;
                        cloud_3d->points.push_back (point);
                    }
                }
            }
        }
        ros::Time now = ros::Time::now();
        pcl_conversions::toPCL(now, cloud_3d->header.stamp);
        cloud_3d->header.frame_id = camera_link_name;
        cloudRGBPublisher.publish(*cloud_3d);




        // std::cout << "pt_3d_h.at<float>(0) is: " << pt_3d_h.at<float>(1,15) << std::endl;
        
        // for all epipolar lines
        // for (vector<cv::Vec3f>::const_iterator it= linesLeft.begin(); it!=linesLeft.end(); ++it) {

        //     // draw the epipolar line between first and last column
        //     cv::line(imgRight,cv::Point(0,-(*it)[2]/(*it)[1]),cv::Point(imgRight.cols,-((*it)[2]+(*it)[0]*imgRight.cols)/(*it)[1]),cv::Scalar(255,255,255));
        // }
    }
    
}

int main(int argc, char *argv[]) {
    
    // Initializing google logging (Caffe uses it as its logging module)
    google::InitGoogleLogging("rtcpm_stereo");
    ROS_INFO("google::InitGoogleLogging");
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    ROS_INFO("gflags::ParseCommandLineFlags");
    // Applying user defined configuration and/or default parameter values to global parameters
    auto return_value = setGlobalParametersFromFlags();
    if (return_value != 0)
        return return_value;
    ROS_INFO("setGlobalParametersFromFlags");
    // Configure frames source
    return_value = readImageDirIfFlagEnabled();
    if (return_value != 0)
        return return_value;
    ROS_INFO("readImageDirIfFlagEnabled");



    // Running rtcpm_stereo
    return_value = rtcpm_stereo();
    if (return_value != 0)
        return return_value;
    ROS_INFO("rtcpm_stereo started");

    ros::init(argc, argv, "ros_rtpose");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    ROS_INFO("ROS node started");

    //ROS starts here
    poseLeftImagePublisher = it.advertise(PUBLISH_LEFT_IMG_TOPIC_NAME, 1);
    poseRightImagePublisher = it.advertise(PUBLISH_RIGHT_IMG_TOPIC_NAME, 1);
    // poseStrPublisher = nh.advertise<std_msgs::String>(PUBLISH_STR_TOPIC_NAME,1000);
    // poseAryPublisher = nh.advertise<std_msgs::Int32MultiArray>(PUBLISH_ARY_TOPIC_NAME, 100);
    // poseAryPublisher = nh.advertise<std_msgs::Float32MultiArray>(PUBLISH_ARY_TOPIC_NAME, 100);
    poseDetectionPublisher = nh.advertise<rtpose_ros::Detection>(PUBLISH_DETECTION_NAME, 10);
    cloudRGBPublisher = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(PUBLISH_3D_PCD_NAME, 1);
    
    // image_transport::Subscriber sub = it.subscribe(RECEIVE_IMG_TOPIC_NAME, 1, getFrameFromROS);
    message_filters::Subscriber<sensor_msgs::Image> left_image_sub(nh, RECEIVE_LEFT_IMG_TOPIC_NAME, 1);
    message_filters::Subscriber<sensor_msgs::Image> right_image_sub(nh, RECEIVE_RIGHT_IMG_TOPIC_NAME, 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync(left_image_sub, right_image_sub, 10);
    sync.registerCallback(boost::bind(&getStereoFrameFromROS, _1, _2));

    // Publish the result stereo images to ROS if images are ready in global.output_queue_ordered
    image_transport::Subscriber sub2 = it.subscribe(RECEIVE_LEFT_IMG_TOPIC_NAME, 1, displayROSFrameStereo); //Actually This is used for publish frame

    // Start to reconstruct the 3d pose by epipolar geometry if any detection result has been generated.
    ros::Subscriber sub3 = nh.subscribe(PUBLISH_DETECTION_NAME,1,reconstruct_3d_pose);
    
    ros::spin();

    ros::shutdown();
    return return_value;
}
