# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import List, Tuple

import cv2
import numpy as np


class TablePreprocess:
    def __init__(self):
        self.max_len = 488

        self.std = np.array([0.229, 0.224, 0.225])
        self.mean = np.array([0.485, 0.456, 0.406])
        self.scale = 1 / 255.0

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        img, shape_list = self.resize_image(img)
        img = self.normalize(img)
        img, shape_list = self.pad_img(img, shape_list)
        img = self.to_chw(img)

        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        return img, shape_list

    def resize_image(self, img: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        h, w = img.shape[:2]
        ratio = self.max_len / (max(h, w) * 1.0)
        resize_h, resize_w = int(h * ratio), int(w * ratio)

        resize_img = cv2.resize(img, (resize_w, resize_h))
        return resize_img, [h, w, ratio, ratio]

    def normalize(self, img: np.ndarray) -> np.ndarray:
        return (img.astype("float32") * self.scale - self.mean) / self.std

    def pad_img(
            self, img: np.ndarray, shape: List[float]
    ) -> Tuple[np.ndarray, List[float]]:
        padding_img = np.zeros((self.max_len, self.max_len, 3), dtype=np.float32)
        h, w = img.shape[:2]
        padding_img[:h, :w, :] = img.copy()
        shape.extend([self.max_len, self.max_len])
        return padding_img, shape

    def to_chw(self, img: np.ndarray) -> np.ndarray:
        return img.transpose((2, 0, 1))


class BatchTablePreprocess:
    """批量表格预处理类"""

    def __init__(self):
        self.preprocess = TablePreprocess()

    def __call__(self, img_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[List[float]]]:
        """批量处理图像

        Args:
            img_list: 图像列表

        Returns:
            预处理后的图像列表和形状信息列表
        """
        if not img_list:
            return None, None

        processed_imgs = []
        shape_lists = []

        for img in img_list:
            if img is None:
                continue

            img_processed, shape_list = self.preprocess.resize_image(img)
            img_processed = self.preprocess.normalize(img_processed)
            img_processed, shape_list = self.preprocess.pad_img(img_processed, shape_list)
            img_processed = self.preprocess.to_chw(img_processed)

            processed_imgs.append(img_processed)
            shape_lists.append(shape_list)

        return processed_imgs, shape_lists
