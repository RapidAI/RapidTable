# -*- encoding: utf-8 -*-
import re
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ...inference_engine.base import get_engine
from ...utils import EngineType
from ..utils import get_struct_str
from .consts import (
    BBOX_TOKENS,
    EOS_TOKEN,
    IMG_SIZE,
    TASK_TOKENS,
    VALID_HTML_BBOX_TOKENS,
)


class UniTableStructure:
    def __init__(self, cfg: Dict[str, Any]):
        if cfg["engine_type"] is None:
            cfg["engine_type"] = EngineType.TORCH
        self.model = get_engine(cfg["engine_type"])(cfg)

        self.encoder = self.model.encoder
        self.device = self.model.device

        self.vocab = self.model.vocab

        self.token_white_list = [
            self.vocab.token_to_id(i) for i in VALID_HTML_BBOX_TOKENS
        ]

        self.bbox_token_ids = set(self.vocab.token_to_id(i) for i in BBOX_TOKENS)
        self.bbox_close_html_token = self.vocab.token_to_id("]</td>")

        self.prefix_token_id = self.vocab.token_to_id("[html+bbox]")

        self.eos_id = self.vocab.token_to_id(EOS_TOKEN)

        self.context = (
            torch.tensor([self.prefix_token_id], dtype=torch.int32)
            .repeat(1, 1)
            .to(self.device)
        )
        self.eos_id_tensor = torch.tensor(self.eos_id, dtype=torch.int32).to(
            self.device
        )

        self.max_seq_len = 1024
        self.img_size = IMG_SIZE
        self.transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.86597056, 0.88463002, 0.87491087],
                    std=[0.20686628, 0.18201602, 0.18485524],
                ),
            ]
        )

        self.decoder = self.model.decoder
        self.decoder.setup_caches(
            max_batch_size=1,
            max_seq_length=self.max_seq_len,
            dtype=torch.float32,
            device=self.device,
        )

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> Tuple[List[str], np.ndarray, float]:
        start_time = time.perf_counter()

        ori_h, ori_w = image.shape[:2]
        image = self.preprocess_img(image)

        memory = self.encoder(image)
        context = self.loop_decode(self.context, self.eos_id_tensor, memory)
        bboxes, html_tokens = self.decode_tokens(context)
        bboxes = self.rescale_bboxes(ori_h, ori_w, bboxes)
        structure_list = get_struct_str(html_tokens)
        elapse = time.perf_counter() - start_time
        return structure_list, bboxes, elapse

    def rescale_bboxes(self, ori_h, ori_w, bboxes):
        scale_h = ori_h / self.img_size
        scale_w = ori_w / self.img_size
        bboxes[:, 0::2] *= scale_w  # 缩放 x 坐标
        bboxes[:, 1::2] *= scale_h  # 缩放 y 坐标
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, ori_w - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, ori_h - 1)
        return bboxes

    def preprocess_img(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image

    def decode_tokens(self, context):
        pred_html = context[0]
        pred_html = pred_html.detach().cpu().numpy()
        pred_html = self.vocab.decode(pred_html, skip_special_tokens=False)
        seq = pred_html.split("<eos>")[0]
        token_black_list = ["<eos>", "<pad>", *TASK_TOKENS]
        for i in token_black_list:
            seq = seq.replace(i, "")

        tr_pattern = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
        td_pattern = re.compile(r"<td(.*?)>(.*?)</td>", re.DOTALL)
        bbox_pattern = re.compile(r"\[ bbox-(\d+) bbox-(\d+) bbox-(\d+) bbox-(\d+) \]")

        decoded_list, bbox_coords = [], []

        # 查找所有的 <tr> 标签
        for tr_match in tr_pattern.finditer(pred_html):
            tr_content = tr_match.group(1)
            decoded_list.append("<tr>")

            # 查找所有的 <td> 标签
            for td_match in td_pattern.finditer(tr_content):
                td_attrs = td_match.group(1).strip()
                td_content = td_match.group(2).strip()
                if td_attrs:
                    decoded_list.append("<td")
                    # 可能同时存在行列合并，需要都添加
                    attrs_list = td_attrs.split()
                    for attr in attrs_list:
                        decoded_list.append(" " + attr)
                    decoded_list.append(">")
                    decoded_list.append("</td>")
                else:
                    decoded_list.append("<td></td>")

                # 查找 bbox 坐标
                bbox_match = bbox_pattern.search(td_content)
                if bbox_match:
                    xmin, ymin, xmax, ymax = map(int, bbox_match.groups())
                    # 将坐标转换为从左上角开始顺时针到左下角的点的坐标
                    coords = np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
                    bbox_coords.append(coords)
                else:
                    # 填充占位的bbox，保证后续流程统一
                    bbox_coords.append(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
            decoded_list.append("</tr>")

        bbox_coords_array = np.array(bbox_coords).astype(np.float32)
        return bbox_coords_array, decoded_list

    def loop_decode(self, context, eos_id_tensor, memory):
        box_token_count = 0
        for _ in range(self.max_seq_len):
            eos_flag = (context == eos_id_tensor).any(dim=1)
            if torch.all(eos_flag):
                break

            next_tokens = self.decoder(memory, context)
            if next_tokens[0] in self.bbox_token_ids:
                box_token_count += 1
                if box_token_count > 4:
                    next_tokens = torch.tensor(
                        [self.bbox_close_html_token], dtype=torch.int32
                    )
                    box_token_count = 0
            context = torch.cat([context, next_tokens], dim=1)
        return context
