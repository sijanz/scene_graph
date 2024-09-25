#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <geometry_msgs/Vector3.h>
#include <tf/transform_datatypes.h>
#include <limits.h>
#include <numeric>
#include <visualization_msgs/MarkerArray.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include "scene_graph/DetectedObjects.h"
#include <std_msgs/String.h>
#include <chrono>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_2_algorithms.h>
#include <string.h>
#include <fstream>

#include "object_location_node.hpp"
#include "scene_graph/GraphObjects.h"

ObjectLocationModule::ObjectLocationModule() : m_nh{}, m_depth_image{}, m_camera_pose{}, m_debug_mode{false}
{
    m_detected_objects_sub = m_nh.subscribe("/scene_graph/detected_objects", 100, &ObjectLocationModule::detectedObjectsCallback, this);

    // TODO: set according to dataset, use parameters
    m_depth_sub = m_nh.subscribe("/camera/depth/pointcloud", 100, &ObjectLocationModule::depthCallback, this);
    m_camera_pose_sub = m_nh.subscribe("/camera/odom", 100, &ObjectLocationModule::cameraPoseCallback, this);
    m_camera_image_sub = m_nh.subscribe("/camera/color/image_raw", 100, &ObjectLocationModule::cameraImageCallback, this);
    m_detected_objects_mseg_sub = m_nh.subscribe("/scene_graph/mseg_segmented_image", 100, &ObjectLocationModule::detectedObjectsMsegCallback, this);
    
    m_graph_objects_pub = m_nh.advertise<scene_graph::GraphObjects>("/scene_graph/seen_graph_objects", 100);

    // visualization publishers
    m_pointcloud_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/scene_graph/viz/current_pointcloud", 10);
    m_seen_object_center_pub = m_nh.advertise<visualization_msgs::MarkerArray>("/scene_graph/viz/current_objects/centers", 10);
    m_segment_pointcloud_pub = m_nh.advertise<sensor_msgs::PointCloud2>("/scene_graph/viz/current_objects/points", 10);

    std::cout << "end of ctor\n";
}

void ObjectLocationModule::detectedObjectsMsegCallback(const sensor_msgs::ImageConstPtr& msg)
{
    std::cout << "in mseg callback\n";
 
    if (m_depth_image == nullptr || m_camera_pose == nullptr)
        return;

    ros::Time start_time = ros::Time::now();

    scene_graph::GraphObjects graph_objects{};
    graph_objects.header.stamp = ros::Time::now();

    cv_bridge::CvImagePtr cv_ptr{};
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    int image_width{cv_ptr->image.size().width};
    int image_height{cv_ptr->image.size().height};

    if (image_width == 0 || image_height == 0 || m_depth_image->data.size() == 0)
        return;

    std::vector<int> detected_classes{};

    for (int y = 0; y < cv_ptr->image.rows; ++y) {
        for (int x = 0; x < cv_ptr->image.cols; ++x) {
            auto value = cv_ptr->image.at<cv::Vec3b>(y, x)[0];

            bool found{false};

            for (auto c : detected_classes) {
                if (c == value) {
                    found = true;
                    break;
                }
            }
            if (!found)
                detected_classes.emplace_back(value);
        }
    }

    // DEBUG
    std::cout << "detected classes:\n";
    for (auto c : detected_classes)
        std::cout << getClassNameFromID(c) << std::endl;


    for (auto detected_class : detected_classes) {

        if (detected_class == 43 || detected_class == 36 || detected_class == 191)
            continue;

        std::vector<geometry_msgs::Point32> pointcloud_segment{};

        double average_depth{0.0};

        for (auto y = 0; y < cv_ptr->image.rows; ++y) {
            for (auto x = 0; x < cv_ptr->image.cols; ++x) {

                if (cv_ptr->image.at<cv::Vec3b>(y, x)[0] == detected_class) {

                    int array_position{y * image_height + x};

                    sensor_msgs::PointCloud2ConstIterator<float> iter_x{*m_depth_image, "x"};
                    sensor_msgs::PointCloud2ConstIterator<float> iter_y{*m_depth_image, "y"};
                    sensor_msgs::PointCloud2ConstIterator<float> iter_z{*m_depth_image, "z"};

                    iter_x += array_position;
                    iter_y += array_position;
                    iter_z += array_position;



                    if (*iter_z > 0.0) {
                        geometry_msgs::Point32 p{};
                        p.x = *iter_x;
                        p.y = *iter_y;
                        p.z = *iter_z;

                        pointcloud_segment.push_back(p);
                    }
                }
            }
        }

        if (pointcloud_segment.size() == 0)
            continue;

        auto center_point{computeMedianPoint(pointcloud_segment)};
        auto filtered_pointcloud_segment{getNearestPoints(pointcloud_segment, center_point)};

        if (filtered_pointcloud_segment.size() == 0)
            continue;

        for (auto& point : filtered_pointcloud_segment)
            point = convertToGlobalPoint(point);

        auto bounding_box{computeBoundingBox(filtered_pointcloud_segment)};

        scene_graph::GraphObject graph_object{};

        std_msgs::String class_name{};
        class_name.data = getClassNameFromID(detected_class);

        graph_object.name = class_name;
        graph_object.bounding_box.push_back(bounding_box.first);
        graph_object.bounding_box.push_back(bounding_box.second);

        graph_objects.objects.push_back(graph_object);
    }

    m_graph_objects_pub.publish(graph_objects);

    // LOG
    ros::Time end_time = ros::Time::now();

    // Calculate the elapsed time
    ros::Duration elapsed_time = end_time - start_time;

    std::ofstream file{};
    file.open("/home/nes/.ros/log/object_location_node.log", std::ios::app);

        if (file.is_open()) {
        file << "[TIMING]: " << std::fixed << std::setprecision(6) << elapsed_time.toSec() << std::endl;

        // Close the file
        file.close();
    } else {
        std::cerr << "Unable to open the file." << std::endl;
    }
}


std::string ObjectLocationModule::getClassNameFromID(int t_id)
{
    if (m_mseg_map.find(t_id) != m_mseg_map.end()) {
        return m_mseg_map[t_id];
    } else {
        return "";
    }
}



void ObjectLocationModule::detectedObjectsCallback(const scene_graph::DetectedObjectsConstPtr& msg)
{
    if (m_depth_image == nullptr || m_camera_pose == nullptr || m_camera_image == nullptr)
        return;

    ros::Time start_time = ros::Time::now();

    scene_graph::GraphObjects graph_objects{};
    graph_objects.header.stamp = ros::Time::now();

    int i = 0;
    for (const auto& detected_object : msg->objects) {
        std::vector<geometry_msgs::Point32> pointcloud_segment{};

        std::chrono::steady_clock::time_point before_loop = std::chrono::steady_clock::now();

        auto delta_y{static_cast<int>(detected_object.bounding_box.at(1).y) - static_cast<int>(detected_object.bounding_box.at(0).y)};
        auto delta_x{static_cast<int>(detected_object.bounding_box.at(1).x) - static_cast<int>(detected_object.bounding_box.at(0).x)};

        for (auto y = static_cast<int>(detected_object.bounding_box.at(0).y); y < static_cast<int>(detected_object.bounding_box.at(1).y); ++y) {
            for (auto x = static_cast<int>(detected_object.bounding_box.at(0).x); x < static_cast<int>(detected_object.bounding_box.at(1).x); ++x) {

                if (!isPointInPolygon(detected_object.segment, x, y))
                    continue;

                int array_position{static_cast<int>(y * m_camera_image->width + x)};

                sensor_msgs::PointCloud2ConstIterator<float> iter_x{*m_depth_image, "x"};
                sensor_msgs::PointCloud2ConstIterator<float> iter_y{*m_depth_image, "y"};
                sensor_msgs::PointCloud2ConstIterator<float> iter_z{*m_depth_image, "z"};

                iter_x += array_position;
                iter_y += array_position;
                iter_z += array_position;

                if (*iter_z > 0.0) {
                    geometry_msgs::Point32 p{};
                    p.x = *iter_x;
                    p.y = *iter_y;
                    p.z = *iter_z;

                    pointcloud_segment.push_back(p);
                }
            }
        }


        if (pointcloud_segment.size() == 0)
            continue;

        auto center_point{computeMedianPoint(pointcloud_segment)};
        auto filtered_pointcloud_segment{getNearestPoints(pointcloud_segment, center_point)};

        if (filtered_pointcloud_segment.size() == 0)
            continue;

        for (auto& point : filtered_pointcloud_segment)
            point = convertToGlobalPoint(point);

        auto bounding_box{computeBoundingBox(filtered_pointcloud_segment)};

        scene_graph::GraphObject graph_object{};
        graph_object.name = detected_object.class_name;
        graph_object.bounding_box.push_back(bounding_box.first);
        graph_object.bounding_box.push_back(bounding_box.second);

        graph_objects.objects.push_back(graph_object);

        ++i;
    }


    m_graph_objects_pub.publish(graph_objects);

    // LOG
    ros::Time end_time = ros::Time::now();

    // Calculate the elapsed time
    ros::Duration elapsed_time = end_time - start_time;

    std::ofstream file{};
    file.open("/home/nes/.ros/log/object_location_node.log", std::ios::app);

        if (file.is_open()) {
        file << "[TIMING]: " << std::fixed << std::setprecision(6) << elapsed_time.toSec() << std::endl;

        // Close the file
        file.close();
    } else {
        std::cerr << "Unable to open the file." << std::endl;
    }
}

void ObjectLocationModule::depthCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    m_depth_image = msg;
}


void ObjectLocationModule::cameraPoseCallback(const nav_msgs::OdometryConstPtr& msg)
{
    m_camera_pose = msg;
}

void ObjectLocationModule::cameraImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    m_camera_image = msg;
}


geometry_msgs::Point32& ObjectLocationModule::convertToGlobalPoint(geometry_msgs::Point32& t_point)
{
    geometry_msgs::Point32 local_point{t_point};

    // TODO: this offset is needed for the dataset - remove if needed!
    // Define a 90 degree rotation quaternion around the z-axis
    tf2::Quaternion q_0;
    q_0.setRPY(0, M_PI_2, 0);  // Roll = 0, Pitch = 0, Yaw = 90 degrees (M_PI/2 radians)

    // // Convert the quaternion to a rotation matrix
    tf2::Matrix3x3 rotation_matrix_0(q_0);

    float x = t_point.x;
    float y = t_point.y;
    float z = t_point.z;

    // Rotate the point around the z-axis
    local_point.x = rotation_matrix_0[0][0] * x + rotation_matrix_0[0][1] * y + rotation_matrix_0[0][2] * z;
    local_point.y = rotation_matrix_0[1][0] * x + rotation_matrix_0[1][1] * y + rotation_matrix_0[1][2] * z;
    local_point.z = rotation_matrix_0[2][0] * x + rotation_matrix_0[2][1] * y + rotation_matrix_0[2][2] * z;

    // --- end of offset calculation

    geometry_msgs::Point32 global_point{};

    tf2::Quaternion q{m_camera_pose->pose.pose.orientation.x, m_camera_pose->pose.pose.orientation.y, m_camera_pose->pose.pose.orientation.z, m_camera_pose->pose.pose.orientation.w};
    tf2::Matrix3x3 rotation_matrix{q};

    // Rotate the local point by the robot's orientation
    tf2::Vector3 local_point_tf(local_point.x, local_point.y, local_point.z);
    tf2::Vector3 rotated_point = rotation_matrix * local_point_tf;

    // Translate the rotated point by the robot's global position
    
    global_point.x = rotated_point.x() + m_camera_pose->pose.pose.position.x;
    global_point.y = rotated_point.y() + m_camera_pose->pose.pose.position.y;
    global_point.z = rotated_point.z() + m_camera_pose->pose.pose.position.z;

    t_point = global_point;

    return t_point;
}


double ObjectLocationModule::yawFromPose(const geometry_msgs::Pose& t_pose)
{
    auto q{tf::Quaternion{t_pose.orientation.x, t_pose.orientation.y, t_pose.orientation.z,
                            t_pose.orientation.w}};
    auto m{tf::Matrix3x3{q}};

    double roll{0.0}, pitch{0.0}, theta{0.0};
    m.getRPY(roll, pitch, theta);

    return theta;
}


visualization_msgs::Marker ObjectLocationModule::createMarkerFromPoint(const geometry_msgs::Point32& t_point, int t_i) const
{
     // Create a Marker message
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";  // Set the frame_id, adjust as necessary
    marker.header.stamp = ros::Time::now();

    marker.ns = "sphere_namespace";  // Namespace for grouping related markers
    marker.id = t_i;  // Unique ID for the marker in this namespace
    marker.type = visualization_msgs::Marker::SPHERE;  // Set marker type to SPHERE
    marker.action = visualization_msgs::Marker::ADD;  // Action to add or modify the marker

    marker.pose.position.x = t_point.x;
    marker.pose.position.y = t_point.y;
    marker.pose.position.z = t_point.z;

    // Set the orientation of the sphere (no rotation needed, so use default orientation)
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    // Set the scale of the sphere (diameter in x, y, z dimensions)
    marker.scale.x = 0.2;  // Example size of the sphere (adjust as needed)
    marker.scale.y = 0.2;
    marker.scale.z = 0.2;

    // Set the color and transparency of the sphere
    marker.color.r = 0.0;  // Red color
    marker.color.g = 0.0;  // Green color
    marker.color.b = 1.0;  // Blue color
    marker.color.a = 1.0;  // Alpha (transparency), 1.0 is fully opaque

    return marker;
}


std::pair<geometry_msgs::Point32, geometry_msgs::Point32> ObjectLocationModule::computeBoundingBox(const std::vector<geometry_msgs::Point32>& points) const
{
    if (points.empty()) {
        throw std::invalid_argument("The input vector of points is empty");
    }

    // Initialize min and max points with the first point in the vector
    geometry_msgs::Point32 min_point = points[0];
    geometry_msgs::Point32 max_point = points[0];

    // Iterate through all points to find the min and max for each coordinate
    for (const auto& point : points) {
        min_point.x = std::min(min_point.x, point.x);
        min_point.y = std::min(min_point.y, point.y);
        min_point.z = std::min(min_point.z, point.z);

        max_point.x = std::max(max_point.x, point.x);
        max_point.y = std::max(max_point.y, point.y);
        max_point.z = std::max(max_point.z, point.z);
    }

    // Return a pair containing the min and max points of the bounding box
    return std::make_pair(min_point, max_point);
}


float ObjectLocationModule::computeMedian(std::vector<float>& values) const {
    size_t n = values.size();
    
    if (n == 0) {
        throw std::invalid_argument("Cannot compute median of an empty vector");
    }

    std::sort(values.begin(), values.end());

    if (n % 2 == 1) {
        // Odd number of elements
        return values[n / 2];
    } else {
        // Even number of elements
        return (values[(n / 2) - 1] + values[n / 2]) / 2.0;
    }
}


geometry_msgs::Point32 ObjectLocationModule::computeMedianPoint(const std::vector<geometry_msgs::Point32>& points) const
{
    if (points.empty()) {
        throw std::invalid_argument("The input vector of points is empty");
    }

    std::vector<float> x_values, y_values, z_values;

    // Extract the x, y, and z values from points
    for (const auto& point : points) {
        x_values.push_back(point.x);
        y_values.push_back(point.y);
        z_values.push_back(point.z);
    }

    // Compute the median for each coordinate
    float median_x = computeMedian(x_values);
    float median_y = computeMedian(y_values);
    float median_z = computeMedian(z_values);

    // Create and return the median point
    geometry_msgs::Point32 median_point;
    median_point.x = median_x;
    median_point.y = median_y;
    median_point.z = median_z;

    return median_point;
}


sensor_msgs::PointCloud2 ObjectLocationModule::createPointCloud2FromPoints(const std::vector<geometry_msgs::Point32>& points) const
{
     sensor_msgs::PointCloud2 cloud_msg;
    
    // Fill in the PointCloud2 header
    cloud_msg.header.stamp = ros::Time::now();
    cloud_msg.header.frame_id = "map";  // You can change this to the desired frame

    // Define the PointCloud2 fields (x, y, z)
    cloud_msg.height = 1;  // Unordered point cloud
    cloud_msg.width = points.size();
    cloud_msg.is_dense = false;  // Set to false if there are invalid points
    
    // Define fields: x, y, z
    cloud_msg.fields.resize(3);
    
    cloud_msg.fields[0].name = "x";
    cloud_msg.fields[0].offset = 0;
    cloud_msg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    cloud_msg.fields[0].count = 1;
    
    cloud_msg.fields[1].name = "y";
    cloud_msg.fields[1].offset = 4;
    cloud_msg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    cloud_msg.fields[1].count = 1;
    
    cloud_msg.fields[2].name = "z";
    cloud_msg.fields[2].offset = 8;
    cloud_msg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    cloud_msg.fields[2].count = 1;
    
    // Set point step and row step
    cloud_msg.point_step = 12;  // 4 bytes * 3 fields (x, y, z)
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
    
    // Resize the data vector to accommodate all points
    cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);
    
    // Fill in the PointCloud2 data
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

    for (size_t i = 0; i < points.size(); ++i, ++iter_x, ++iter_y, ++iter_z)
    {
        *iter_x = points[i].x;
        *iter_y = points[i].y;
        *iter_z = points[i].z;
    }

    return cloud_msg;
}


bool ObjectLocationModule::isPointOnLine(int x, int y, int x1, int y1, int x2, int y2) const
{
    if (x <= std::max(x1, x2) && x >= std::min(x1, x2) &&
        y <= std::max(y1, y2) && y >= std::min(y1, y2)) {
        return true;
    }
    return false;
}


int ObjectLocationModule::orientation(int px, int py, int qx, int qy, int rx, int ry) const
{
    int val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy);
    if (val == 0) return 0;  // collinear
    return (val > 0) ? 1 : 2;  // clock or counterclock wise
}


bool ObjectLocationModule::doIntersect(int p1x, int p1y, int q1x, int q1y, int p2x, int p2y, int q2x, int q2y) const
{
    int o1 = orientation(p1x, p1y, q1x, q1y, p2x, p2y);
    int o2 = orientation(p1x, p1y, q1x, q1y, q2x, q2y);
    int o3 = orientation(p2x, p2y, q2x, q2y, p1x, p1y);
    int o4 = orientation(p2x, p2y, q2x, q2y, q1x, q1y);

    if (o1 != o2 && o3 != o4) return true;

    // Special Cases
    if (o1 == 0 && isPointOnLine(p2x, p2y, p1x, p1y, q1x, q1y)) return true;
    if (o2 == 0 && isPointOnLine(q2x, q2y, p1x, p1y, q1x, q1y)) return true;
    if (o3 == 0 && isPointOnLine(p1x, p1y, p2x, p2y, q2x, q2y)) return true;
    if (o4 == 0 && isPointOnLine(q1x, q1y, p2x, p2y, q2x, q2y)) return true;

    return false;
}


bool ObjectLocationModule::isPointInPolygon(const std::vector<geometry_msgs::Point32>& t_polygon, float t_px, float t_py) const
{
       // Define the kernel
    typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

    // Define the types for points and polygon
    typedef K::Point_2 Point;
    typedef CGAL::Polygon_2<K> Polygon;

    // Define a polygon
    Polygon polygon{};

    for (auto& point : t_polygon)
        polygon.push_back(Point(point.x, point.y));

    Point point{t_px, t_py};

    // Check if the point is inside the polygon
    CGAL::Bounded_side result = CGAL::bounded_side_2(polygon.vertices_begin(), polygon.vertices_end(), point, K());

    return result == CGAL::ON_BOUNDED_SIDE || result == CGAL::ON_BOUNDARY ? true : false;
}


double ObjectLocationModule::calculateDistance(const geometry_msgs::Point32& p1, const geometry_msgs::Point32& p2) const
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
}


std::vector<geometry_msgs::Point32> ObjectLocationModule::getNearestPoints(const std::vector<geometry_msgs::Point32>& points, const geometry_msgs::Point32& center) const
{
     // Create a vector to store distances along with corresponding points
    std::vector<std::pair<double, geometry_msgs::Point32>> distances;

    // Calculate the distance of each point from the center and store it
    for (const auto& point : points) {
        double distance = calculateDistance(point, center);
        distances.push_back(std::make_pair(distance, point));
    }

    // Sort the distances vector based on the distances
    std::sort(distances.begin(), distances.end(), [](const std::pair<double, geometry_msgs::Point32>& a, const std::pair<double, geometry_msgs::Point32>& b) {
        return a.first < b.first;
    });

    // Get the number of points that represent the closest 75%
    int num_points_to_select = static_cast<int>(distances.size() * 0.70);

    // Extract the closest 50% points
    std::vector<geometry_msgs::Point32> nearest_points;
    for (size_t i = 0; i < num_points_to_select; ++i) {
        nearest_points.push_back(distances[i].second);
    }

    return nearest_points;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_location_module");
    ObjectLocationModule op{};
    ros::spin();
    return 0;
}
