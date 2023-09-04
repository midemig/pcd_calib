# pcd_calib


## Ussage

Launch theoretical tf tree in ros

Get theoretical transformation between source and target LiDAR:

    rosrun tf tf_echo frame_target_lidar frame_source_lidar

Change pcd_calib.launch initial x, y, z, roll, pitch, yaw with this data.
Change target and source lidar topic and target_frame
Configure SLAM method with target LiDAR topic

Launch calibration (one terminal for each command):

    roscore
    rosparam set /use_sim_time true
    rosbag play _2023-06-06-14* /tf:=/filter_tf /tf_static:=/filter_static_tf --pause --clock -r 0.4 -s 0
    roslaunch hdl_graph_slam hdl_graph_slam_test.launch
    roslaunch pcd_calib pcd_calib_f150_3.launch

Copy results (line "Final" in terminal) to the python script