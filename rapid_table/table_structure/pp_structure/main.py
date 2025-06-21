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

from rapid_table.utils.typings import EngineType

from ...inference_engine.base import get_engine
from .post_process import TableLabelDecode
from .pre_process import TablePreprocess


class PPTableStructurer:
    def __init__(self, cfg: Dict[str, Any]):
        if cfg["engine_type"] is None:
            cfg["engine_type"] = EngineType.ONNXRUNTIME
        self.session = get_engine(cfg["engine_type"])(cfg)

        self.preprocess_op = TablePreprocess()

        self.character = self.session.get_character_list()
        self.postprocess_op = TableLabelDecode(self.character)

    def __call__(self, img: np.ndarray) -> Tuple[List[str], np.ndarray, float]:
        s = time.perf_counter()

        img, shape_list = self.preprocess_op(img)

        bbox_preds, struct_probs = self.session(img.copy())

        post_result = self.postprocess_op(bbox_preds, struct_probs, [shape_list])
        table_struct_str = self.get_struct_str(post_result)
        bbox_list = post_result["bbox_batch_list"][0]

        elapse = time.perf_counter() - s
        return table_struct_str, bbox_list, elapse

    def get_struct_str(self, post_result: Dict[str, Any]) -> List[str]:
        structure_str_list = post_result["structure_batch_list"][0][0]
        structure_str_list = (
            ["<html>", "<body>", "<table>"]
            + structure_str_list
            + ["</table>", "</body>", "</html>"]
        )
        return structure_str_list
