#include <ros/ros.h>
#include <thread>

// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl/common/transforms.h>
#include <pcl_ros/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/conditional_removal.h>

#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <vector>
#include <cfloat>
#include <iostream>
#include <fstream>

#define n_clouds 50
// int n_clouds = 50;


void PCDCalib(ros::NodeHandle *nh);
void get_params(ros::NodeHandle *nh);
void tarjet_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input_cloud);
void source_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input_cloud);
void quat2euler(float qx, float qy, float qz, float qw, double *roll, double *pitch, double *yaw);
void iterate();
Eigen::Matrix<float, 4, 4> tf_matrix(float x, float y, float z, tf::Transform transform);
std::vector<double> icp_full(float x, float y, float z, float qx, float qy, float qz, float qw, float grid_size);
std::vector<double> icp(float x, float y, float z, float qx, float qy, float qz, float qw);
static void thread_icp(int i, Eigen::Matrix<float, 4, 4> initial_guess, Eigen::Matrix<float, n_clouds, 8> *results_array);
static void publish_cloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int pub_id);
static inline float SIGN(float x);
static inline float NORM(float a, float b, float c, float d);
static std::vector<double> mRot2Quat(const Eigen::Matrix4d& matrix);
void print_results(float x, float y, float z, tf2::Quaternion q, float score);
float quat_diff(tf2::Quaternion q);
float distance(float x, float y, float z);


float tarjet_x = -1.0;
float tarjet_y = 0.0;
float tarjet_z = -0.4;
float tarjet_roll = 0.0;
float tarjet_pitch = -0.7;
float tarjet_yaw =  0.0;


pcl::PointCloud<pcl::PointXYZI>::Ptr target_full_cloud(new pcl::PointCloud<pcl::PointXYZI>);


static ros::Publisher pub, pub2, pub3;
std::vector<double> BoundingBoxesAngle;


std::vector<pcl::PointCloud<pcl::PointXYZI>> tarjet_cloud_array;
static std::vector<pcl::PointCloud<pcl::PointXYZI>> source_cloud_array;

static std::vector<tf::StampedTransform> transform_array;

tf::TransformListener* listener;
tf::TransformBroadcaster* br;

ros::Subscriber tarjet_cloud_sub, source_cloud_sub;

std::string target_cloud_topic, source_cloud_topic, map_frame, sensor_frame;

float initial_x, initial_y, initial_z, initial_roll, initial_pitch, initial_yaw;

float min_x, min_y, min_z;

bool first_source_cloud, first_target_cloud;
double last_target_time, last_source_time;
float target_time_interval, source_time_interval;

pcl::ConditionOr<pcl::PointXYZI>::Ptr range_cond (new pcl::ConditionOr<pcl::PointXYZI> ());
pcl::ConditionalRemoval<pcl::PointXYZI> condrem;

// std::ofstream myfile;
// myfile.open("/home/mimiguel/bagfiles/Calibracion/data.txt");


void PCDCalib(ros::NodeHandle *nh) {

  ROS_INFO("START!");


  listener = new tf::TransformListener();
  br = new tf::TransformBroadcaster();

  get_params(nh);

  // Create a ROS subscriber for the input point cloud
  tarjet_cloud_sub = nh->subscribe (target_cloud_topic, 1, &tarjet_cloud_cb);
  source_cloud_sub = nh->subscribe (source_cloud_topic, 1, &source_cloud_cb);

  // Create a ROS publisher for the output model coefficients
  pub = nh->advertise<sensor_msgs::PointCloud2> ("output", 1);
  pub2 = nh->advertise<sensor_msgs::PointCloud2> ("output2", 1);
  pub3 = nh->advertise<sensor_msgs::PointCloud2> ("output3", 1);

  first_source_cloud = first_target_cloud = true;


  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZI>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZI> ("x", pcl::ComparisonOps::LT, -min_x)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZI>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZI> ("x", pcl::ComparisonOps::GT, min_x)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZI>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZI> ("y", pcl::ComparisonOps::LT, -min_y)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZI>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZI> ("y", pcl::ComparisonOps::GT, min_y)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZI>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZI> ("z", pcl::ComparisonOps::LT, -min_z)));
  range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZI>::ConstPtr (new pcl::FieldComparison<pcl::PointXYZI> ("z", pcl::ComparisonOps::GT, min_z)));

}

void get_params(ros::NodeHandle *nh)
{

  nh->param<std::string>("target_cloud_topic", target_cloud_topic, "/ada/lidar32_center/pointcloud");
  nh->param<std::string>("source_cloud_topic", source_cloud_topic, "/ada/lidar16_right/pointcloud");
  
  nh->param<std::string>("map_frame", map_frame, "map");
  nh->param<std::string>("sensor_frame", sensor_frame, "ada_lidar32_center_link");
  
  nh->param<float>("initial_x", initial_x, -0.075707);
  nh->param<float>("initial_y", initial_y, 1.118574);
  nh->param<float>("initial_z", initial_z, -0.597746);
  nh->param<float>("initial_roll", initial_roll, -0.000942);
  nh->param<float>("initial_pitch", initial_pitch, 0.785782);
  nh->param<float>("initial_yaw", initial_yaw, 0.725210);

  nh->param<float>("target_time_interval", target_time_interval, 1.0);
  nh->param<float>("source_time_interval", source_time_interval, 1.0);
  
  nh->param<float>("min_x", min_x, 1.5);
  nh->param<float>("min_y", min_y, 1.5);
  nh->param<float>("min_z", min_z, 1.5);
  
  // nh->param<int>("n_clouds", n_clouds, 50);

}

void tarjet_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input_cloud)
{
  double cloud_time = (*input_cloud).header.stamp.sec + double((*input_cloud).header.stamp.nsec) / double(1000000000);


  if(!first_target_cloud && (cloud_time - last_target_time) < target_time_interval)
    return;
  
  first_target_cloud = false;
  last_target_time = cloud_time;

  pcl::PointCloud<pcl::PointXYZI>::Ptr aux_cloud(new pcl::PointCloud<pcl::PointXYZI>);

  tf::StampedTransform transform;
  try
  {
    listener->lookupTransform(map_frame, sensor_frame, ros::Time(0), transform);
  }
  catch (tf::TransformException ex){
    ROS_ERROR("%s",ex.what());
    ros::Duration(1.0).sleep();
    return;
  }


  pcl::fromROSMsg(*input_cloud, *aux_cloud);

  condrem.setCondition (range_cond);
  condrem.setInputCloud (aux_cloud);
  condrem.setKeepOrganized(true);
  condrem.filter (*aux_cloud);

  pcl_ros::transformPointCloud(*aux_cloud, *aux_cloud, transform);

  tarjet_cloud_array.push_back(*aux_cloud);

  if(source_cloud_array.size()>=n_clouds)
  {
    tarjet_cloud_sub.shutdown();
    source_cloud_sub.shutdown();
    iterate();
  }
}

void source_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input_cloud)
{
  double cloud_time = (*input_cloud).header.stamp.sec + double((*input_cloud).header.stamp.nsec) / double(1000000000);

  if(!first_source_cloud && (cloud_time - last_source_time) < source_time_interval)
    return;
  
  first_source_cloud = false;
  last_source_time = cloud_time;

  pcl::PointCloud<pcl::PointXYZI>::Ptr aux_cloud(new pcl::PointCloud<pcl::PointXYZI>);

  tf::StampedTransform transform;
  try
  {
    listener->lookupTransform(map_frame, sensor_frame, ros::Time(0), transform);
  }
  catch (tf::TransformException ex)
  {
    ROS_ERROR("%s",ex.what());
    ros::Duration(1.0).sleep();
    return;
  }

  pcl::fromROSMsg(*input_cloud, *aux_cloud);

  condrem.setCondition (range_cond);
  condrem.setInputCloud (aux_cloud);
  condrem.setKeepOrganized(true);
  condrem.filter (*aux_cloud);

  source_cloud_array.push_back(*aux_cloud);

  if(source_cloud_array.size() > n_clouds)
  {
    source_cloud_array.erase(source_cloud_array.begin());
  }

  transform_array.push_back(transform);

  if(transform_array.size() > n_clouds)
  {
    transform_array.erase(transform_array.begin());
  }      

}

void quat2euler(float qx, float qy, float qz, float qw, double *roll, double *pitch, double *yaw)
{
  tf::Quaternion q(qx, qy, qz, qw);
  tf::Matrix3x3 m(q);
  m.getRPY(*roll, *pitch, *yaw);
}

void print_results(float x, float y, float z, tf2::Quaternion q, float score)
{
  double roll, pitch, yaw;

  quat2euler(q[0], q[1], q[2], q[3], &roll, &pitch, &yaw);
  printf("%f\t %f\t %f\t %f\t %f\t %f\t\n", x, y, z, roll, pitch, yaw);
  printf("\t\tError t: %f - error r: %f - score: %f\n", distance(x, y, z), quat_diff(q), score);

}


float quat_diff(tf2::Quaternion q)
{
  tf2::Quaternion tarjet_q;
  tarjet_q.setRPY(tarjet_roll, tarjet_pitch, tarjet_yaw);
  float w = q[3]*tarjet_q[3] + q[0]*tarjet_q[0] + q[1]*tarjet_q[1] + q[2]*tarjet_q[2];
  float i = q[3]*tarjet_q[0] - q[0]*tarjet_q[3] - q[1]*tarjet_q[2] + q[2]*tarjet_q[1];
  float j = q[3]*tarjet_q[1] + q[0]*tarjet_q[2] - q[1]*tarjet_q[3] - q[2]*tarjet_q[0];
  float k = q[3]*tarjet_q[2] - q[0]*tarjet_q[1] + q[1]*tarjet_q[0] - q[2]*tarjet_q[3];
  return atan2(sqrt(pow(i, 2) + pow(j, 2) + pow(k, 2)), w);
}


float distance(float x, float y, float z)
{
  return sqrt(pow(x - tarjet_x, 2) + pow(y - tarjet_y, 2) + pow(z - tarjet_z, 2));
}


void iterate()
{

  for(int i = 0; i<tarjet_cloud_array.size() ; i++)
  {
    *target_full_cloud += tarjet_cloud_array[i];
  }

  std::vector<double> transform_guess;

  tf2::Quaternion q; 
  q.setRPY(initial_roll, initial_pitch, initial_yaw);

  double roll, pitch, yaw;
  quat2euler(q[0], q[1], q[2], q[3], &roll, &pitch, &yaw);
  printf("Initial %f\t %f\t %f\t %f\t %f\t %f\t \n", initial_x, initial_y, initial_z, roll, pitch, yaw);

  transform_guess = icp(initial_x, initial_y, initial_z, q[0], q[1], q[2], q[3]);
  printf("It 0: ");
  print_results(transform_guess[0], transform_guess[1], transform_guess[2], tf2::Quaternion(transform_guess[3], transform_guess[4], transform_guess[5], transform_guess[6]), transform_guess[7]);

  for(int i=0; i<20; i++)
  {
    transform_guess = icp_full(transform_guess[0], transform_guess[1], transform_guess[2], transform_guess[3], transform_guess[4], transform_guess[5], transform_guess[6], 0.3);
    printf("It %d a: ", i);
    print_results(transform_guess[0], transform_guess[1], transform_guess[2], tf2::Quaternion(transform_guess[3], transform_guess[4], transform_guess[5], transform_guess[6]), transform_guess[7]);

    transform_guess = icp(transform_guess[0], transform_guess[1], transform_guess[2], transform_guess[3], transform_guess[4], transform_guess[5], transform_guess[6]);
    printf("It %d b: ", i);
    print_results(transform_guess[0], transform_guess[1], transform_guess[2], tf2::Quaternion(transform_guess[3], transform_guess[4], transform_guess[5], transform_guess[6]), transform_guess[7]);

  }


  // transform_guess = icp_full(transform_guess[0], transform_guess[1], transform_guess[2], transform_guess[3], transform_guess[4], transform_guess[5], transform_guess[6], 0.3);
  // quat2euler(transform_guess[3], transform_guess[4], transform_guess[5], transform_guess[6], &roll, &pitch, &yaw);
  printf("Final: ");
  print_results(transform_guess[0], transform_guess[1], transform_guess[2], tf2::Quaternion(transform_guess[3], transform_guess[4], transform_guess[5], transform_guess[6]), transform_guess[7]);

  // myfile.close();

  ros::spinOnce();
  ros::spinOnce();

  exit(0);
}

Eigen::Matrix<float, 4, 4> tf_matrix(float x, float y, float z, tf::Transform transform)
{
  tf::Matrix3x3 rotation_matrix;
  Eigen::Matrix<float, 4, 4> initial_guess;

  rotation_matrix = transform.getBasis();

  for(int i = 0;i < 4;i++)
  {
    for(int j = 0;j < 4;j++)
    {
      initial_guess(i,j) = rotation_matrix[i][j];
    }
  }

  initial_guess(0,3) = x;
  initial_guess(1,3) = y;
  initial_guess(2,3) = z;
  initial_guess(3,3) = 1;

  initial_guess(3,0) = 0;
  initial_guess(3,1) = 0;
  initial_guess(3,2) = 0;

  return initial_guess;

}



std::vector<double> icp_full(float x, float y, float z, float qx, float qy, float qz, float qw, float grid_size)
{
  
  printf("\tFull cloud iteration\n");


  pcl::PointCloud<pcl::PointXYZI>::Ptr source_full_cloud (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_local_cloud  (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_aligned (new pcl::PointCloud<pcl::PointXYZI>);
  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();
  std::vector<double> transform_result;

  sensor_msgs::PointCloud2 send_cloud;


  tf2::Quaternion q(qx, qy, qz, qw);
  tf::Transform transform;
  tf::Vector3 v(x, y, z);
  tf::Quaternion quat(q[0], q[1], q[2], q[3]);
  transform.setOrigin(v);
  transform.setRotation(quat);


  // ROS_INFO("Source size: %d\n", source_cloud_array.size());
  for(int i = 0; i<source_cloud_array.size() ; i++)
  {
    pcl_ros::transformPointCloud(source_cloud_array[i], *source_local_cloud, transform);
    pcl_ros::transformPointCloud(*source_local_cloud, *source_local_cloud, transform_array[i]);
    *source_full_cloud += *source_local_cloud;
  }
  pcl_ros::transformPointCloud(*source_full_cloud, *source_full_cloud, transform.inverse());

  Eigen::Matrix<float, 4, 4> initial_guess = tf_matrix(x, y, z, transform);

  pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud (source_full_cloud);
  sor.setLeafSize (grid_size, grid_size, grid_size);
  sor.filter (*source_full_cloud);

  sor.setInputCloud (target_full_cloud);
  sor.setLeafSize (grid_size, grid_size, grid_size);
  sor.filter (*target_full_cloud);

  pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
  icp.setMaximumIterations (50);
  icp.setMaxCorrespondenceDistance (0.5);
  icp.setRANSACOutlierRejectionThreshold (0.01);
  icp.setTransformationEpsilon (1e-9);
  icp.setInputSource (source_full_cloud);
  icp.setInputTarget (target_full_cloud);
  icp.align (*cloud_aligned, initial_guess);

  if (icp.hasConverged ())
  {
    // std::cout << "\nICP has converged, score is " << icp.getFitnessScore () << std::endl;
    transformation_matrix = icp.getFinalTransformation ().cast<double>();
    transform_result = mRot2Quat(transformation_matrix);
    transform_result.push_back(icp.getFitnessScore());
  }


  transform.setOrigin(tf::Vector3(transform_result[0], transform_result[1], transform_result[2]));
  transform.setRotation(tf::Quaternion(transform_result[3], transform_result[4], transform_result[5], transform_result[6]));


  // pcl_ros::transformPointCloud(*cloud_4, *cloud_4, transform);

  publish_cloud(target_full_cloud, 1);
  publish_cloud(source_full_cloud, 2);
  publish_cloud(cloud_aligned, 3);

  return transform_result;
}


std::vector<double> icp(float x, float y, float z, float qx, float qy, float qz, float qw)
{
  printf ("\tMulti PCD Iteration :\n");

  pcl::PointCloud<pcl::PointXYZI>::Ptr source_full_cloud (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_local_cloud  (new pcl::PointCloud<pcl::PointXYZI>);

  tf2::Quaternion q(qx, qy, qz, qw);
  tf::Transform transform;
  tf::Vector3 v(x, y, z);
  tf::Quaternion quat(q[0], q[1], q[2], q[3]);
  transform.setOrigin(v);
  transform.setRotation(quat);

  Eigen::Matrix<float, n_clouds, 8> results_array;
  std::vector<double> transform_result;
  std::vector<std::thread> thread_array;

  Eigen::Matrix<float, 4, 4> initial_guess = tf_matrix(x, y, z, transform);

  for(int i = 0; i<n_clouds ; i++)
  {
    thread_array.push_back(std::thread(&thread_icp, i, initial_guess, &results_array));
    // std::thread t(thread_icp, i, initial_guess, &results_array);
    // thread_array.push_back(t);
  }

  for(int i = 0; i<thread_array.size() ; i++)
    thread_array[i].join();

  std::cout << "Column's maximum: " << results_array.colwise().maxCoeff() << std::endl;
  std::cout << "Column's minimum: " << results_array.colwise().minCoeff() << std::endl;
  std::cout << "Column's mean   : " << results_array.colwise().mean() << std::endl;

  // myfile << "results_array.colwise().mean()";
  // myfile << "\n";

  auto mean_values = results_array.colwise().mean();

  for(int c= 0; c< 8; c++)
    transform_result.push_back(mean_values[c]);

  publish_cloud(target_full_cloud, 1);


  transform.setOrigin(tf::Vector3(mean_values[0], mean_values[1], mean_values[2]));
  transform.setRotation(tf::Quaternion(mean_values[3], mean_values[4], mean_values[5], mean_values[6]));


  for(int i = 0; i<source_cloud_array.size() ; i++)
  {
    pcl_ros::transformPointCloud(source_cloud_array[i], *source_local_cloud, transform);
    pcl_ros::transformPointCloud(*source_local_cloud, *source_local_cloud, transform_array[i]);
    *source_full_cloud += *source_local_cloud;
  }

  publish_cloud(source_full_cloud, 3);

  // printf("\tmean_values: %f\t %f\t %f\t %f\t %f\t %f\t\n", mean_values[0], mean_values[1], mean_values[2], mean_values[3], mean_values[4], mean_values[5]);

  return transform_result;

}

static void thread_icp(int i, Eigen::Matrix<float, 4, 4> initial_guess, Eigen::Matrix<float, n_clouds, 8> *results_array)
{

  pcl::PointCloud<pcl::PointXYZI>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud_transformed (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_aligned (new pcl::PointCloud<pcl::PointXYZI>);
  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();
  std::vector<double> transform_result;

  pcl_ros::transformPointCloud(*target_full_cloud, *target_cloud_transformed, transform_array[i].inverse());

  *source_cloud = source_cloud_array[i];

  // publish_cloud(source_cloud, 3);
  // publish_cloud(target_cloud_transformed, 1);

  pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
  icp.setMaximumIterations (5);
  icp.setMaxCorrespondenceDistance (0.5);
  icp.setRANSACOutlierRejectionThreshold (0.01);
  icp.setTransformationEpsilon (1e-9);
  icp.setInputSource (source_cloud);
  icp.setInputTarget (target_cloud_transformed);
  icp.align (*cloud_aligned, initial_guess);

  if (icp.hasConverged ())
  {
    transformation_matrix = icp.getFinalTransformation ().cast<double>();
    transform_result = mRot2Quat(transformation_matrix);
    transform_result.push_back(0);//icp.getFitnessScore());
  }

  for(int c= 0; c< 8; c++)
    (*results_array)(i, c) = transform_result[c];
}


static void publish_cloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int pub_id)
{
  sensor_msgs::PointCloud2 send_cloud;

  pcl::toROSMsg(*cloud, send_cloud);
  send_cloud.header.frame_id = "map";

  switch(pub_id){
    case 1:
      pub.publish(send_cloud);

    case 2:
      pub2.publish(send_cloud);

    case 3:
      pub3.publish(send_cloud);
  }

}


static inline float SIGN(float x) { 
  return (x >= 0.0f) ? +1.0f : -1.0f; 
}

static inline float NORM(float a, float b, float c, float d) { 
  return sqrt(a * a + b * b + c * c + d * d); 
}

// quaternion = [w, x, y, z]'
static std::vector<double> mRot2Quat(const Eigen::Matrix4d& matrix) {
  float r11 = matrix(0, 0);
  float r12 = matrix(0, 1);
  float r13 = matrix(0, 2);
  float r21 = matrix(1, 0);
  float r22 = matrix(1, 1);
  float r23 = matrix(1, 2);
  float r31 = matrix(2, 0);
  float r32 = matrix(2, 1);
  float r33 = matrix(2, 2);
  float q0 = (r11 + r22 + r33 + 1.0f) / 4.0f;
  float q1 = (r11 - r22 - r33 + 1.0f) / 4.0f;
  float q2 = (-r11 + r22 - r33 + 1.0f) / 4.0f;
  float q3 = (-r11 - r22 + r33 + 1.0f) / 4.0f;
  if (q0 < 0.0f) {
    q0 = 0.0f;
  }
  if (q1 < 0.0f) {
    q1 = 0.0f;
  }
  if (q2 < 0.0f) {
    q2 = 0.0f;
  }
  if (q3 < 0.0f) {
    q3 = 0.0f;
  }
  q0 = sqrt(q0);
  q1 = sqrt(q1);
  q2 = sqrt(q2);
  q3 = sqrt(q3);
  if (q0 >= q1 && q0 >= q2 && q0 >= q3) {
    q0 *= +1.0f;
    q1 *= SIGN(r32 - r23);
    q2 *= SIGN(r13 - r31);
    q3 *= SIGN(r21 - r12);
  }
  else if (q1 >= q0 && q1 >= q2 && q1 >= q3) {
    q0 *= SIGN(r32 - r23);
    q1 *= +1.0f;
    q2 *= SIGN(r21 + r12);
    q3 *= SIGN(r13 + r31);
  }
  else if (q2 >= q0 && q2 >= q1 && q2 >= q3) {
    q0 *= SIGN(r13 - r31);
    q1 *= SIGN(r21 + r12);
    q2 *= +1.0f;
    q3 *= SIGN(r32 + r23);
  }
  else if (q3 >= q0 && q3 >= q1 && q3 >= q2) {
    q0 *= SIGN(r21 - r12);
    q1 *= SIGN(r31 + r13);
    q2 *= SIGN(r32 + r23);
    q3 *= +1.0f;
  }
  else {
    printf("coding error\n");
  }
  float r = NORM(q0, q1, q2, q3);
  q0 /= r;
  q1 /= r;
  q2 /= r;
  q3 /= r;

  std::vector<double> transform;

  transform.push_back(matrix(0, 3));
  transform.push_back(matrix(1, 3));
  transform.push_back(matrix(2, 3));
  transform.push_back(q1);
  transform.push_back(q2);
  transform.push_back(q3);
  transform.push_back(q0);
  // [x, y, z, w]
  return transform;
}


int main (int argc, char **argv)
{
  ros::init(argc, argv, "pcl_cluster");
  ros::NodeHandle nh("~");
  // PCDCalib c = PCDCalib(&nh);
  PCDCalib(&nh);
  ros::spin();
}
