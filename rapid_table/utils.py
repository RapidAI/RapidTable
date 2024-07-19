# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

InputType = Union[str, np.ndarray, bytes, Path]


class LoadImage:
    def __init__(
        self,
    ):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        img = self.load_img(img)

        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3 and img.shape[2] == 4:
            return self.cvt_four_to_three(img)

        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = np.array(Image.open(img))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = np.array(Image.open(BytesIO(img)))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        if isinstance(img, np.ndarray):
            return img

        raise LoadImageError(f"{type(img)} is not supported!")

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → RGB"""
        r, g, b, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))

        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(new_img, new_img, mask=a)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


class LoadImageError(Exception):
    pass


class VisTable:
    def __init__(
        self,
    ):
        self.load_img = LoadImage()

    def __call__(
        self,
        img_path: Union[str, Path],
        table_html_str: str,
        save_html_path: Optional[str] = None,
        table_cell_bboxes: Optional[np.ndarray] = None,
        save_drawed_path: Optional[str] = None,
    ) -> None:
        if save_html_path:
            html_with_border = self.insert_border_style(table_html_str)
            self.save_html(save_html_path, html_with_border)

        if table_cell_bboxes is None:
            return None

        img = self.load_img(img_path)

        dims_bboxes = table_cell_bboxes.shape[1]
        if dims_bboxes == 4:
            drawed_img = self.draw_rectangle(img, table_cell_bboxes)
        elif dims_bboxes == 8:
            drawed_img = self.draw_polylines(img, table_cell_bboxes)
        else:
            raise ValueError("Shape of table bounding boxes is not between in 4 or 8.")

        if save_drawed_path:
            self.save_img(save_drawed_path, drawed_img)

        return drawed_img

    def insert_border_style(self, table_html_str: str):
        style_res = """<style>td {border-left: 1px solid;border-bottom:1px solid;}
                    table, th {border-top:1px solid;font-size: 10px;
                    border-collapse: collapse;border-right: 1px solid;}
                    </style>"""
        prefix_table, suffix_table = table_html_str.split("<body>")
        html_with_border = f"{prefix_table}{style_res}<body>{suffix_table}"
        return html_with_border

    @staticmethod
    def draw_rectangle(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        img_copy = img.copy()
        for box in boxes.astype(int):
            x1, y1, x2, y2 = box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return img_copy

    @staticmethod
    def draw_polylines(img: np.ndarray, points) -> np.ndarray:
        img_copy = img.copy()
        for point in points.astype(int):
            point = point.reshape(4, 2)
            cv2.polylines(img_copy, [point.astype(int)], True, (255, 0, 0), 2)
        return img_copy

    @staticmethod
    def save_img(save_path: Union[str, Path], img: np.ndarray):
        cv2.imwrite(str(save_path), img)

    @staticmethod
    def save_html(save_path: Union[str, Path], html: str):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
