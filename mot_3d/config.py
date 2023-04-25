""" TypedDict classes to represent the configuration options stored as yaml 
files in `configs/<dataset>_configs/*.yaml`
"""

from typing import TypedDict, Literal


class RunningConfig(TypedDict):
    covariance: Literal["default"]
    score_threshold: float
    max_age_since_update: int
    min_hits_to_birth: int
    match_type: Literal["bipartite"]
    nms_threshold: float | None
    asso: Literal["iou", "giou"]
    has_velo: bool
    motion_model: Literal["kf", "fbkf"]
    asso_thres: TypedDict("AssoThreshold", {"giou": float, "iou": float})


class RedundancyConfig(TypedDict):
    mode: Literal["default"]
    det_score_threshold: TypedDict(
        "DetScoreThreshold", {"iou": float, "giou": float, "m_dis": float}
    )
    det_dist_threshold: TypedDict(
        "DetDistThreshold", {"iou": float, "giou": float, "m_dis": float}, total=False
    )


class DataLoaderConfig(TypedDict, total=False):
    pc: bool
    nms: bool | None
    nms_threshold: float | None


class Config(TypedDict):
    running: RunningConfig
    redundancy: RedundancyConfig
    data_loader: DataLoaderConfig


config: Config = {
    "running": {
        "covariance": "default",
        "score_threshold": 0.01,
        "max_age_since_update": 2,
        "min_hits_to_birth": 1,
        "match_type": "bipartite",
        "asso": "iou",
        "has_velo": False,
        "motion_model": "kf",
        "asso_thres": {"giou": 1.5, "iou": 0.9},
    },
    "redundancy": {
        "mode": "default",
        "det_score_threshold": {"iou": 0.01, "giou": 0.01, "m_dis": 0.01},
        "det_dist_threshold": {"iou": 0.1, "giou": -0.5},
    },
    "data_loader": {"pc": False},
}
