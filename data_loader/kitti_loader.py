import os
import json
from pathlib import Path
from typing import Dict, Any, List, TypedDict
from typing import List, Tuple, Optional

import numpy as np

import mot_3d.utils as utils
from mot_3d.data_protos import BBox
from mot_3d.preprocessing import nms


class DatasetSample(TypedDict):
    # The time stamp of the current frame, in seconds
    time_stamp: float

    # The ego vehicle's information at the current frame, including
    # position, orientation as a numpy array of shape (4, 4)
    ego: np.ndarray

    # The types of the detected objects in the current frame, as a list of integers.
    # Each integer corresponds to a specific object type (e.g., car, pedestrian, etc.).
    det_types: List[int]

    # The bounding boxes of the detected objects in the current frame
    dets: List[BBox]

    # The point cloud data of the current frame, as a numpy array of shape (N, 3).
    # If point cloud data is not available, this should be None.
    pc: np.ndarray | None

    # Additional auxiliary information about the current frame, such as whether it is a key frame
    # and the velocities of the detected objects. This is a dictionary with the following keys and
    # value types:
    #   - is_key_frame: a boolean indicating whether the current frame is a key frame
    #   - velos: np.array of shape (2, 1)] ! see velo2world(),
    aux_info: TypedDict(
        "AuxInfo",
        {
            "is_key_frame": bool,
            "velos": np.array,
        },
        total=False,
    )


# kitti/training/{label_2,velodyne}

# types: 'Car' | 'Van' | 'Truck' | 'Pedestrian' | 'Person_sitting' | 'Cyclist' | 'Tram' | 'Misc' | 'DontCare'


class KittiLoader:
    def __init__(
        self,
        configs: Dict[
            str, Any
        ],  # A yaml file. See SimpleTrack/configs/kitti_configs/*.yaml
        type_token: List[int],  # can be vehicle=1, pedestriant=2, cyclist=4
        segment_name: str,  # data/ego_info/segment without npz extension
        data_folder: Path,  # data/
        det_data_folder: Path,  # data/detections/
        start_frame: int
        | None,  # What frame to start at in the sequence of frame. If None, then start at frame 0
    ):
        self.configs = configs
        self.segment = segment_name
        self.data_loader = data_folder
        self.det_data_folder = det_data_folder
        self.type_token = type_token

        self.token_info = json.load(
            open(
                os.path.join(
                    data_folder, "token_info", "{:}.json".format(segment_name)
                ),
                "r",
            )
        )
        self.ego_info = np.load(
            os.path.join(data_folder, "ego_info", "{:}.npz".format(segment_name)),
            allow_pickle=True,
        )
        self.calib_info = np.load(
            os.path.join(data_folder, "calib_info", "{:}.npz".format(segment_name)),
            allow_pickle=True,
        )
        self.dets = np.load(
            os.path.join(det_data_folder, "dets", "{:}.npz".format(segment_name)),
            allow_pickle=True,
        )
        self.det_type_filter = True

        # Frequency = 10Hz
        # Generate list of timestamps at 10Hz with the same length as dets
        self.time_stamps = np.arange(0, self.dets["bboxes"].shape[0], 1 / 10)
        self.is_key_frames = [True for _ in range(self.dets["bboxes"].shape[0])]
        self.ts_info = zip(self.time_stamps, self.is_key_frames)

        self.use_pc = configs["data_loader"]["pc"]
        if self.use_pc:
            self.pcs = np.load(
                os.path.join(
                    data_folder, "pc", "raw_pc", "{:}.npz".format(segment_name)
                ),
                allow_pickle=True,
            )

        self.max_frame = len(self.dets["bboxes"])
        self.selected_frames = [
            i for i in range(self.max_frame) if self.token_info[i][3]
        ]
        self.cur_selected_index = 0
        self.cur_frame = start_frame if start_frame is not None else 0
        assert (
            0 <= self.cur_frame < self.max_frame
        ), f"Start frame {self.cur_frame} is out of range"

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return self.max_frame

    def __next__(self):
        if self.cur_frame >= self.max_frame:
            raise StopIteration
        idx: int = self.cur_frame

        bboxes = self.dets["bboxes"][idx]
        inst_types = self.dets["types"][idx]
        det_types = [
            inst_types[i]
            for i in range(len(bboxes))
            if inst_types[i] in self.type_token
        ]
        selected_dets = [
            bboxes[i] for i in range(len(bboxes)) if inst_types[i] in self.type_token
        ]

        ego = self.ego_info["ego"][str(idx)]

        velos = None
        if "velos" in self.dets.keys():
            cur_frame_velos = self.dets["velos"][self.cur_frame]
            tmp = [
                np.array(cur_frame_velos[i])
                for i in range(len(bboxes))
                if inst_types[i] in self.type_token
            ]
            result["aux_info"]["velos"] = [utils.velo2world(ego, v) for v in tmp]

        pc = None
        if self.use_pc:
            pc = self.pcs[str(idx)]
            pc = utils.pc2world(ego, pc)

        dets = [BBox.bbox2world(ego, BBox.array2bbox(b)) for b in selected_dets]
        if self.nms:
            dets, det_types, velos = self.frame_nms(
                dets, det_types, velos, self.nms_thres
            )
        dets = [BBox.bbox2array(d) for d in dets]

        result: DatasetSample = {
            "time_stamp": self.time_stamps[idx],
            "ego": ego,
            "det_types": det_types,
            "dets": dets,
            "aux_info": {"velos": velos, "is_key_frame": self.is_key_frames[idx]},
            "pc": pc,
        }

        self.cur_frame += 1
        return result

    def frame_nms(
        self,
        dets: List[BBox],
        det_types: List[int],
        velos: Optional[List[float]],
        thres: float,
    ) -> Tuple[List[BBox], List[int], Optional[List[float]]]:
        frame_indexes, frame_types = nms(dets, det_types, thres)
        result_dets = [dets[i] for i in frame_indexes]
        result_velos = None
        if velos is not None:
            result_velos = [velos[i] for i in frame_indexes]
        return result_dets, frame_types, result_velos
