raw_data_dir=$1
data_dir_2hz=$2
data_dir_20hz=$3
version=$4

# token information
python token_info.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz --version $version
python token_info.py --raw_data_folder $raw_data_dir --data_folder $data_dir_20hz --mode 20hz --version $version

# time stamp information
python time_stamp.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz --version $version
python time_stamp.py --raw_data_folder $raw_data_dir --data_folder $data_dir_20hz --mode 20hz --version $version

# sensor calibration information
python sensor_calibration.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz --version $version
python sensor_calibration.py --raw_data_folder $raw_data_dir --data_folder $data_dir_20hz --mode 20hz --version $version

# ego pose
python ego_pose.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz --version $version
python ego_pose.py --raw_data_folder $raw_data_dir --data_folder $data_dir_20hz --mode 20hz --version $version

# gt information
python gt_info.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz --version $version
python gt_info.py --raw_data_folder $raw_data_dir --data_folder $data_dir_20hz --mode 20hz --version $version

# point cloud, useful for visualization
python raw_pc.py --raw_data_folder $raw_data_dir --data_folder $data_dir_2hz --mode 2hz --version $version
python raw_pc.py --raw_data_folder $raw_data_dir --data_folder $data_dir_20hz --mode 20hz --version $version

