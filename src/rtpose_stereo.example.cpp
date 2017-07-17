#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#define USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER 1

#if USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
#  include <image_transport/subscriber_filter.h>
#else
#  include <sensor_msgs/Image.h>
#  include <message_filters/subscriber.h>
#endif

class MyClass {
public:
  MyClass() :
    it_(nh_),
#if USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
    left_image_sub_( it_, "image/orig", 1 ),
    right_image_sub_( it_, "image/warp", 1 ),
#else
    left_image_sub_( nh_, "image/orig", 1 ),
    right_image_sub_( nh_, "image/warp", 1 ),
#endif
    sync( MySyncPolicy( 10 ), left_image_sub_, right_image_sub_ )
  {
    sync.registerCallback( boost::bind( &MyClass::callback, this, _1, _2 ) );
  }

  void callback(
    const sensor_msgs::ImageConstPtr& left_img_msg,
    const sensor_msgs::ImageConstPtr& right_img_msg
  ){
    // your code here
  }

private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;

#if USE_IMAGE_TRANSPORT_SUBSCRIBER_FILTER
  typedef image_transport::SubscriberFilter ImageSubscriber;
#else
  typedef message_filters::Subscriber< sensor_msgs::Image > ImageSubscriber;
#endif

  ImageSubscriber left_image_sub_;
  ImageSubscriber right_image_sub_;

  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, sensor_msgs::Image
  > MySyncPolicy;

  message_filters::Synchronizer< MySyncPolicy > sync;
};

int main(int argc, char** argv) {
  ros::init( argc, argv, "my_node" );
  MyClass mc;

  while( ros::ok() ){
    ros::spin();
  }

  return EXIT_SUCCESS;
}