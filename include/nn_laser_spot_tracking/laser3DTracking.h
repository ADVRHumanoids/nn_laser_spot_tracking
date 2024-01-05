#ifndef __LASER3DTRACKING_H_
#define __LASER3DTRACKING_H_

#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <ddynamic_reconfigure/ddynamic_reconfigure.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <nn_laser_spot_tracking/KeypointImage.h>

#include <Eigen/Dense>

#include <utils/SecondOrderFilter.h>

#include <mutex>
#include <array>


typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

namespace NNLST {

class Laser3DTracking {
    
public:
    Laser3DTracking(ros::NodeHandle* nh, const double& period);
    
    bool isReady();
    
    int run();
    
private:
    ros::NodeHandle* _nh;
    double _period;
    
    std::string _camera_frame;
    std::string _ref_frame ;
    std::string _laser_spot_frame;
    
    double _detection_confidence_threshold;
    double _cloud_detection_max_sec_diff;
    ros::Subscriber _keypoint_sub;
    void keypointSubClbk(const nn_laser_spot_tracking::KeypointImageConstPtr& msg);
    nn_laser_spot_tracking::KeypointImage _keypoint_image;
    
    tf2_ros::TransformBroadcaster _tf_broadcaster;
    std::vector<geometry_msgs::TransformStamped> _ref_T_spot; //one for raw, other for filtered. tf2_ros wants vector, cant use std::array

    ros::Subscriber _cloud_sub;
    void cloudClbk(const PointCloud::ConstPtr& msg);
    PointCloud::Ptr _cloud;  
    std::mutex _cloud_mutex;

    /***************************************************** */

    bool sendTransformFrom2D();
    bool updateTransform();
    
    /***************  FILTER    **********/
    NNLST::utils::FilterWrap<Eigen::Vector3d>::Ptr _laser_pos_filter;
    double _filter_damping, _filter_bw;
    
    std::unique_ptr<ddynamic_reconfigure::DDynamicReconfigure> _ddr_server;
    void ddr_callback_filter_damping(double new_value);
    void ddr_callback_filter_bw(double new_value);
    
};

}

#endif // __LASER3DTRACKING_H_
