import os, numpy as np, nuscenes, argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from copy import deepcopy
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='../../../raw/nuscenes/data/sets/nuscenes/')
parser.add_argument('--data_folder', type=str, default='../../../datasets/nuscenes/')
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-test'])
args = parser.parse_args()


def main(nusc, scene_names, root_path, ego_folder, mode):
    pbar = tqdm(total=len(scene_names))
    for scene_index, scene_info in enumerate(nusc.scene):
        scene_name = scene_info['name']
        if scene_name not in scene_names:
            continue
        first_sample_token = scene_info['first_sample_token']
        last_sample_token = scene_info['last_sample_token']
        frame_data = nusc.get('sample', first_sample_token)
        if args.mode == '20hz':
            cur_sample_token = frame_data['data']['LIDAR_TOP']
        elif args.mode == '2hz':
            cur_sample_token = deepcopy(first_sample_token)
        frame_index = 0
        ego_data = dict()
        while True:
            if mode == '2hz':
                frame_data = nusc.get('sample', cur_sample_token)
                lidar_token = frame_data['data']['LIDAR_TOP']
                lidar_data = nusc.get('sample_data', lidar_token)
                ego_token = lidar_data['ego_pose_token']
                ego_pose = nusc.get('ego_pose', ego_token)
            elif mode == '20hz':
                frame_data = nusc.get('sample_data', cur_sample_token)
                ego_token = frame_data['ego_pose_token']
                ego_pose = nusc.get('ego_pose', ego_token)

            # translation + rotation
            ego_data[str(frame_index)] = ego_pose['translation'] + ego_pose['rotation']

            # clean up and prepare for the next
            cur_sample_token = frame_data['next']
            if cur_sample_token == '':
                break
            frame_index += 1
        
        np.savez_compressed(os.path.join(ego_folder, '{:}.npz'.format(scene_name)), **ego_data)
        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    print('ego info')
    ego_folder = os.path.join(args.data_folder, 'ego_info')
    os.makedirs(ego_folder, exist_ok=True)

    if args.version == 'v1.0-test':
        val_scene_names = splits.create_splits_scenes()['test']
    else:
        val_scene_names = splits.create_splits_scenes()['val']
    nusc = NuScenes(version=args.version, dataroot=args.raw_data_folder, verbose=True)
    main(nusc, val_scene_names, args.raw_data_folder, ego_folder, args.mode)
