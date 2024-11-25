# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import copy
import importlib
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .table_matcher import TableMatch
from .table_structure import TableStructurer
from .utils import LoadImage, VisTable

root_dir = Path(__file__).resolve().parent


class RapidTable:
    def __init__(self, model_path: Optional[str] = None, model_type: str = None, use_cuda: bool = False):
        if model_path is None:
            model_path = str(
                root_dir / "models" / "slanet-plus.onnx"
            )
            model_type = "slanet-plus"
        self.model_type = model_type
        self.load_img = LoadImage()
        config = {
            "model_path": model_path,
            "use_cuda": use_cuda,
        }
        self.table_structure = TableStructurer(config)
        self.table_matcher = TableMatch()

        try:
            self.ocr_engine = importlib.import_module("rapidocr_onnxruntime").RapidOCR()
        except ModuleNotFoundError:
            self.ocr_engine = None

    def __call__(
        self,
        img_content: Union[str, np.ndarray, bytes, Path],
        ocr_result: List[Union[List[List[float]], str, str]] = None,
        return_logic_points = False
    ) -> Tuple[str, float]:
        if self.ocr_engine is None and ocr_result is None:
            raise ValueError(
                "One of two conditions must be met: ocr_result is not empty, or rapidocr_onnxruntime is installed."
            )

        img = self.load_img(img_content)

        s = time.time()
        h, w = img.shape[:2]

        if ocr_result is None:
            ocr_result, _ = self.ocr_engine(img)
        dt_boxes, rec_res = self.get_boxes_recs(ocr_result, h, w)

        pred_structures, pred_bboxes, _ = self.table_structure(copy.deepcopy(img))
        # 适配slanet-plus模型输出的box缩放还原
        if self.model_type == "slanet-plus":
            pred_bboxes = self.adapt_slanet_plus(img, pred_bboxes)
        pred_html = self.table_matcher(pred_structures, pred_bboxes, dt_boxes, rec_res)
        # 避免低版本升级后出现问题,默认不打开
        if return_logic_points:
            logic_points = self.table_matcher.decode_logic_points(pred_structures)
            elapse = time.time() - s
            return pred_html, pred_bboxes, logic_points, elapse
        elapse = time.time() - s
        return pred_html, pred_bboxes, elapse

    def get_boxes_recs(
        self, ocr_result: List[Union[List[List[float]], str, str]], h: int, w: int
    ) -> Tuple[np.ndarray, Tuple[str, str]]:
        dt_boxes, rec_res, scores = list(zip(*ocr_result))
        rec_res = list(zip(rec_res, scores))

        r_boxes = []
        for box in dt_boxes:
            box = np.array(box)
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        return dt_boxes, rec_res
    def adapt_slanet_plus(self, img: np.ndarray, pred_bboxes: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        resized = 488
        ratio = min(resized / h, resized / w)
        w_ratio = resized / (w * ratio)
        h_ratio = resized / (h * ratio)
        pred_bboxes[:, 0::2] *= w_ratio
        pred_bboxes[:, 1::2] *= h_ratio
        return pred_bboxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        help="Wheter to visualize the layout results.",
    )
    parser.add_argument(
        "-img", "--img_path", type=str, required=True, help="Path to image for layout."
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default=str(root_dir / "models" / "en_ppstructure_mobile_v2_SLANet.onnx"),
        help="The model path used for inference.",
    )
    args = parser.parse_args()

    try:
        ocr_engine = importlib.import_module("rapidocr_onnxruntime").RapidOCR()
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Please install the rapidocr_onnxruntime by pip install rapidocr_onnxruntime."
        ) from exc

    rapid_table = RapidTable(args.model_path)

    img = cv2.imread(args.img_path)

    ocr_result, _ = ocr_engine(img)
    table_html_str, table_cell_bboxes, elapse = rapid_table(img, ocr_result)
    print(table_html_str)

    viser = VisTable()
    if args.vis:
        img_path = Path(args.img_path)

        save_dir = img_path.resolve().parent
        save_html_path = save_dir / f"{Path(img_path).stem}.html"
        save_drawed_path = save_dir / f"vis_{Path(img_path).name}"
        viser(
            img_path,
            table_html_str,
            save_html_path,
            table_cell_bboxes,
            save_drawed_path,
        )


if __name__ == "__main__":
    main()
