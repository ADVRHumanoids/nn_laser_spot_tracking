#include <nn_laser_spot_tracking/laser3DTracking.h>

using NNLST::Laser3DTracking;

Laser3DTracking::Laser3DTracking (ros::NodeHandle* nh, const double& period)
{
    
    this->_nh = nh;
    this->_period = period;
        
    // Input params
    std::string pc_topic;
    if (_nh->getParam("point_cloud_topic", pc_topic)) {
        throw std::runtime_error("Error: point_cloud_topic param not set");
    }

    // Tracking Params
    _nh->param<double>("detection_confidence_threshold", _detection_confidence_threshold, 0.55);
    _nh->param<double>("cloud_detection_max_sec_diff", _cloud_detection_max_sec_diff, 4);

    // Output params
    std::string keypoint_topic;
    _nh->param<std::string>("keypoint_topic", keypoint_topic, "/nn_laser_spot_tracking/detection_output_keypoint");
    _nh->param<std::string>("laser_spot_frame", _laser_spot_frame, "laser_spot");
    
    _keypoint_sub = _nh->subscribe<nn_laser_spot_tracking::KeypointImage>(keypoint_topic, 10, &Laser3DTracking::keypointSubClbk, this);
     
    /******************* CLOUD ***************************/
    _cloud_sub = _nh->subscribe<PointCloud>(pc_topic, 1, &Laser3DTracking::cloudClbk, this);
    _cloud = boost::make_shared<PointCloud>();

    _ref_T_spot.resize(2);
    _ref_T_spot.at(0).child_frame_id = _laser_spot_frame;
    _ref_T_spot.at(0).transform.rotation.w = 1;

    _ref_T_spot.at(1).child_frame_id = _laser_spot_frame + "_raw";
    _ref_T_spot.at(1).transform.rotation.w = 1;

    
    /************************************************ FILTER  ***************************/
    
    _nh->param<double>("damping", _filter_damping, 1);
    _nh->param<double>("bw", _filter_bw, 9);

    _laser_pos_filter = std::make_shared<NNLST::utils::FilterWrap<Eigen::Vector3d>>(_filter_damping, _filter_bw, _period, 3);
        
    _ddr_server = std::make_unique<ddynamic_reconfigure::DDynamicReconfigure>(*nh);
    _ddr_server->registerVariable<double>("damping", _filter_damping, boost::bind(&Laser3DTracking::ddr_callback_filter_damping, this, _1), "damping", (double)0, (double)10, "laser_filter");
    _ddr_server->registerVariable<double>("bw", _filter_bw, boost::bind(&Laser3DTracking::ddr_callback_filter_bw, this, _1), "bw", (double)0, (double)50, "laser_filter");
    _ddr_server->registerVariable<double>("detection_confidence_threshold", &_detection_confidence_threshold, "Under this confidence (coming from 2d pixel image) the point is considered invalid, and no relative tf is published", 0, 1, "detection");
    _ddr_server->registerVariable<double>("cloud_detection_max_sec_diff", &_cloud_detection_max_sec_diff, "If point cloud and detection keypoint have a timestamp with difference bigger than this value, no relative tf is published", 0, 10, "detection");
    _ddr_server->publishServicesTopics();
    
}

bool Laser3DTracking::isReady() {
    
    if (_cloud->size() == 0) {
        ROS_WARN_STREAM_ONCE("Point cloud not yet arrived...");
        return false;
    } else {
        ROS_WARN_STREAM_ONCE("... Point cloud arrived");
    }
    
    if (_keypoint_sub.getNumPublishers() < 1) {
        ROS_WARN_STREAM_ONCE("Nobody is publishing the 2d keypoints on '"<< _keypoint_sub.getTopic() << "'...");
        return false;
    } else {
        ROS_WARN_STREAM_ONCE("Someone is publishing the 2d keypoints on '"<< _keypoint_sub.getTopic() << "'...");

    }
    
    ROS_INFO_STREAM("Ready!");
    return true;
}

int Laser3DTracking::run () {
    
    if (! sendTransformFrom2D()) {
        return -1;
    }

    return 0;
}

bool Laser3DTracking::sendTransformFrom2D() {
    
    {
    const std::lock_guard<std::mutex> lock(_cloud_mutex);

    //header in pcl cloud is a uint in microsecond, not a ros::Time
    ros::Time cloud_time;
    cloud_time.fromNSec(_cloud->header.stamp*1000);
    //ROS_INFO("cloud time %s;\timage time %s", 
    //        std::to_string(cloud_time.toSec()).c_str(),
    //        std::to_string(keypoint_image.header.stamp.toSec()).c_str());
    
    
    if (_keypoint_image.confidence <= _detection_confidence_threshold ){
    
        //ROS_WARN("Confidence of arrived keypoint detection message is below the threshold (%s < %s)", 
        //         std::to_string(keypoint_image.confidence).c_str(), std::to_string(detection_confidence_threshold).c_str());
        return false;
    }
    
    ros::Duration time_diff;
    
//     std::cout << "timestamps" << std::endl;
//     std::cout << " ros: " << cloud_time.toNSec() << std::endl;
//     std::cout << " pcl: " << cloud->header.stamp * 1000 << std::endl;
//     std::cout << "timestamps" << std::endl;

    if (_keypoint_image.header.stamp < cloud_time) {
        
        time_diff = cloud_time - _keypoint_image.header.stamp;
        
        if (time_diff.toSec() > _cloud_detection_max_sec_diff) {
            ROS_WARN("keypoint is too old wrt to cloud: (%ss, %ss, diff : %ss > %ss)", 
                    std::to_string(_keypoint_image.header.stamp.toSec()).c_str(), 
                    std::to_string(cloud_time.toSec()).c_str(),
                    std::to_string(time_diff.toSec()).c_str(),
                    std::to_string(_cloud_detection_max_sec_diff).c_str()
                    );    
            
            return false;
        } 
        
    } else {
        
        time_diff = _keypoint_image.header.stamp - cloud_time; 
        
        if (time_diff.toSec() > _cloud_detection_max_sec_diff) {
            ROS_WARN("cloud is too old wrt to keypoint (this is strange): (%ss, %ss, diff : %ss > %ss)", 
                    std::to_string(cloud_time.toSec()).c_str(), 
                    std::to_string(_keypoint_image.header.stamp.toSec()).c_str(),
                    std::to_string(time_diff.toSec()).c_str(),
                    std::to_string(_cloud_detection_max_sec_diff).c_str()
                    );    

            return false;
        } 
    }
    
    //std::cout << "timestamp diff: " << time_diff.toSec() << std::endl;
    
    if (!updateTransform()){
        return false;
    }
    
    } // destroy the cloud_mutex;
    
    _tf_broadcaster.sendTransform(_ref_T_spot);
 
    return true;
}

bool Laser3DTracking::updateTransform ()
{
    auto pointXYZ = _cloud->at(_keypoint_image.x_pixel, _keypoint_image.y_pixel);
    
    if (pointXYZ.z == 0 || std::isnan(pointXYZ.x) || std::isnan(pointXYZ.y) || std::isnan(pointXYZ.z) ) {
    
//         ROS_ERROR("Z distance is very small, do not ignore this: x:%f, y:%f, z:%f", pointXYZ.x, pointXYZ.y, pointXYZ.z);
//         ROS_ERROR("x_pix:%d, y_pix%d", _keypoint_image.x_pixel, _keypoint_image.y_pixel);
//         ROS_ERROR("cloud index from pixels:%d\n", _keypoint_image.x_pixel * _cloud->width + _keypoint_image.y_pixel);
        ROS_WARN("pixel has no corresponding pc (probably there is a hole, or the cloud has been filtered out), dropping it");
        return false;
    }
    //ROS_INFO_STREAM(pointXYZ.x << " " << pointXYZ.y << " " << pointXYZ.z);
    
    _ref_T_spot.at(0).header.stamp = ros::Time::now();
    _ref_T_spot.at(1).header.stamp = _ref_T_spot.at(0).header.stamp;
    
    _ref_T_spot.at(0).header.frame_id = _cloud->header.frame_id;
    _ref_T_spot.at(1).header.frame_id = _cloud->header.frame_id;
    
    Eigen::Vector3d vec, vec_filt;
    vec << pointXYZ.x, pointXYZ.y, pointXYZ.z;
    vec_filt = _laser_pos_filter->process(vec);
    
    _ref_T_spot.at(0).transform.translation.x = vec_filt(0);
    _ref_T_spot.at(0).transform.translation.y = vec_filt(1);
    _ref_T_spot.at(0).transform.translation.z = vec_filt(2);
    _ref_T_spot.at(1).transform.translation.x = pointXYZ.x;
    _ref_T_spot.at(1).transform.translation.y = pointXYZ.y;
    _ref_T_spot.at(1).transform.translation.z = pointXYZ.z;
    
    return true;
}

void Laser3DTracking::cloudClbk(const PointCloud::ConstPtr& msg)
{
    const std::lock_guard<std::mutex> lock(_cloud_mutex);
    *_cloud = *msg;
    //std::cout << cloud->header.frame_id << std::endl;
    //change reference frame
    //pcl_ros::transformPointCloud (ref_frame, *cloud, *cloud, tf_buffer);
}

void Laser3DTracking::keypointSubClbk(const nn_laser_spot_tracking::KeypointImageConstPtr& msg)
{
    _keypoint_image = *msg;
}

void Laser3DTracking::ddr_callback_filter_damping(double new_value) {
    
    _filter_damping = new_value;
    _laser_pos_filter->reset(_filter_damping, _filter_bw);
}
void Laser3DTracking::ddr_callback_filter_bw(double new_value) {
    
    _filter_bw = new_value;
    _laser_pos_filter->reset(_filter_damping, _filter_bw);

}

/**
void Laser3DTracking::cameraInfoClbk(const sensor_msgs::CameraInfoConstPtr& msg) {
    
    cam_info = msg;
    
//     disp2Depth = cv::Mat::zeros(4, 4, CV_64FC1);
//     
//     disp2Depth.at<double>(0,0) = 1;
//     disp2Depth.at<double>(1,1) = 1;
//     
//     disp2Depth.at<double>(0,3) = - msg->K.at(3); //K is stored row-major
//     disp2Depth.at<double>(1,3) = - msg->K.at(5); //K is stored row-major
//     disp2Depth.at<double>(2,3) = - msg->K.at(0); //fx is the same of fy???
    
     //P is stored row-major
//     disp2Depth.at<double>(0,3) = - msg->K.at(2); //cx
//     disp2Depth.at<double>(1,3) = - msg->P.at(5); //cy
//     disp2Depth.at<double>(2,3) = - msg->P.at(0); //fx is the same of fy???
//     
//     disp2Depth.at<double>(2,3) = - 1.0 / (msg->P.at(3)); // -1/Tx
//     disp2Depth.at<double>(2,3) = - 1.0 / (msg->P.at(3)); //(cx - cx') / Tx
    
    camera_info_sub.shutdown();
}
**/


/******************************** *****************************************/

int main ( int argc, char **argv ) {

    ros::init ( argc, argv, "3D_matching" );
    ros::NodeHandle nh("~");
    
    double rate;
    nh.param<double>("rate", rate, 100);
    
    Laser3DTracking laser3DTracking(&nh, 1.0/rate);
    
    ros::Rate r0(100);
    while(ros::ok()) {
        
        if (laser3DTracking.isReady()) {
            break;
        }

        ros::spinOnce();
        r0.sleep();
    }
    
    
    ros::Rate r(rate);
    while(ros::ok()) {
        
        laser3DTracking.run();

        ros::spinOnce();
        r.sleep();
    }
    
    return 0;
    
}
