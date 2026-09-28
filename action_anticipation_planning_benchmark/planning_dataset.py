import torch.utils.data as data

import os
import os.path
import numpy as np
from numpy.random import randint
import torch
import sys
from typing import Optional, Set, Dict, Any

# Ensure repo root is on sys.path so `clego_cl` is importable even if cwd is this folder.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clego_cl.task_map import normalize_video_id, get_task_id

from colorama import init
from colorama import Fore, Back, Style

init(autoreset=True)


class VideoRecord(object):

    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]


    @property
    def end_sec(self):
        return float(self._data[1])

    @property
    def labels(self):
        return eval(self._data[2])


class TSNDataSet(data.Dataset):

    def __init__(self,
                 root_path,
                 list_file,
                 feat_path,
                 num_dataload,
                 feat_suffix=".pt",
                 num_segments=3,
                 new_length=1,
                 modality='RGB',
                 image_tmpl='img_{:05d}.t7',
                 transform=None,
                 force_grayscale=False,
                 random_shift=False,
                 test_mode=False,
                 allowed_video_ids: Optional[Set[str]] = None,
                 video_to_task: Optional[Dict[str, int]] = None,
                 unknown_tracker: Optional[Any] = None):

        self.root_path = root_path
        self.feat_path = feat_path
        self.feat_suffix = feat_suffix
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_dataload = num_dataload
        self.use_context_sec = 2
        self.context_stride = 1
        self.feature_fps = 5
        self.num_classes = 27
        self.future_length = 8
        self.feature_dict = dict()
        self.data_type = "anti"
        self.allowed_video_ids = allowed_video_ids
        self.video_to_task = video_to_task
        self.unknown_tracker = unknown_tracker

        if self.modality == 'RGBDiff' or self.modality == 'RGBDiff2' or self.modality == 'RGBDiffplus':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()  # read all the video files

    def _parse_list(self):
        raw_list = [VideoRecord(x.strip().split('|')) for x in open(self.list_file)]

        # Filter by allowed ids (normalized) + track unknown
        if self.allowed_video_ids is not None or self.video_to_task is not None:
            filtered = []
            for rec in raw_list:
                vid = normalize_video_id(rec.path)
                if self.allowed_video_ids is not None and vid not in self.allowed_video_ids:
                    continue
                if self.video_to_task is not None:
                    tid = get_task_id(self.video_to_task, vid)
                    if tid is None and self.unknown_tracker is not None:
                        try:
                            self.unknown_tracker.add(vid)
                        except Exception:
                            pass
                filtered.append(rec)
            raw_list = filtered

        self.video_list = raw_list
        # repeat the list if the length is less than num_dataload (especially for target data)
        if len(self.video_list) == 0:
            return
        n_repeat = self.num_dataload // len(self.video_list)
        n_left = self.num_dataload % len(self.video_list)
        self.video_list = self.video_list * n_repeat + self.video_list[:n_left]


    def get_anti_item(self, index):
        record = self.video_list[index]

        if record.path not in self.feature_dict.keys():
            self.feature_dict[record.path] = torch.load(os.path.join(self.feat_path, record.path + self.feat_suffix))
        
        feature = self.feature_dict[record.path]
        feature_len = feature.shape[0]
        # print("feature_shape: {}".format(feature.shape))
        context_end_sec = record.end_sec

        context_start_sec = context_end_sec - self.use_context_sec
        if context_start_sec < 0:
            return torch.zeros((self.num_segments, 768)), -torch.ones(
                (self.future_length)).long()
        context_start_frame = int(context_start_sec * self.feature_fps)
        context_end_frame = int(context_end_sec * self.feature_fps)
        assert context_start_frame < feature_len and context_end_frame < feature_len, "video path: {} feature_len: {} context_start_frame: {} context_end_frame: {} context_start_sec: {} context_end_sec: {}".format(
            record.path, feature_len, context_start_frame, context_end_frame,
            context_start_sec, context_end_sec)
        context_start_frame = max(0, context_start_frame)
        context_end_frame = min(feature_len - 1, context_end_frame)

        sample_feature = feature[context_start_frame:context_end_frame +
                                 1:self.context_stride]
        if sample_feature.shape[0] != self.num_segments:
            sample_feature = sample_feature.transpose(0,
                                                      1).unsqueeze(0)  # B C T
            sample_feature = torch.nn.functional.interpolate(
                sample_feature, size=self.num_segments,
                mode='linear').squeeze(0).transpose(0, 1)

        # print(sample_feature.shape)
        # if sample_feature.shape[0] > 20:
        #     print(
        #         "video path: {} feature_len: {} context_start_frame: {} context_end_frame: {} context_start_sec: {} context_end_sec: {}"
        #         .format(record.path, feature_len, context_start_frame,
        #                 context_end_frame, context_start_sec, context_end_sec))
        #     exit()

        labels = record.labels
        labels = torch.LongTensor(labels)
        return sample_feature, labels

    def __getitem__(self, index):
        assert self.data_type in ["anti"] # only support long-term anticipation
        if self.data_type == "anti":
            return self.get_anti_item(index)
        else:
            raise ValueError("data_type must be rec or anti")

    def __len__(self):
        return len(self.video_list)
