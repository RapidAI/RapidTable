# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from rapid_table.utils.typings import EngineType, ModelType

from ...inference_engine.base import get_engine
from ..utils import get_struct_str
from .post_process import TableLabelDecode
from .pre_process import TablePreprocess


class PPTableStructurer:
    def __init__(self, cfg: Dict[str, Any]):
        if cfg["engine_type"] is None:
            cfg["engine_type"] = EngineType.ONNXRUNTIME
        self.session = get_engine(cfg["engine_type"])(cfg)
        self.cfg = cfg

        self.preprocess_op = TablePreprocess()

        self.character = self.session.get_character_list()
        self.postprocess_op = TableLabelDecode(self.character)

    def __call__(self, ori_img: np.ndarray) -> Tuple[List[str], np.ndarray, float]:
        s = time.perf_counter()

        img, shape_list = self.preprocess_op(ori_img)

        bbox_preds, struct_probs = self.session(img.copy())

        post_result = self.postprocess_op(bbox_preds, struct_probs, [shape_list])

        table_struct_str = get_struct_str(post_result["structure_batch_list"][0][0])
        cell_bboxes = post_result["bbox_batch_list"][0]

        if self.cfg["model_type"] == ModelType.SLANETPLUS:
            cell_bboxes = self.rescale_cell_bboxes(ori_img, cell_bboxes)
        cell_bboxes = self.filter_blank_bbox(cell_bboxes)

        elapse = time.perf_counter() - s
        return table_struct_str, cell_bboxes, elapse

    def rescale_cell_bboxes(
        self, img: np.ndarray, cell_bboxes: np.ndarray
    ) -> np.ndarray:
        h, w = img.shape[:2]
        resized = 488
        ratio = min(resized / h, resized / w)
        w_ratio = resized / (w * ratio)
        h_ratio = resized / (h * ratio)
        cell_bboxes[:, 0::2] *= w_ratio
        cell_bboxes[:, 1::2] *= h_ratio
        return cell_bboxes

    @staticmethod
    def filter_blank_bbox(cell_bboxes: np.ndarray) -> np.ndarray:
        # 过滤掉占位的bbox
        mask = ~np.all(cell_bboxes == 0, axis=1)
        return cell_bboxes[mask]
