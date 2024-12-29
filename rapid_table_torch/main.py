# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import copy
import importlib
import os
import time
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np

from .download_model import DownloadModel
from .logger import get_logger
from .table_matcher import TableMatch
from .table_structure import TableStructurer
from .utils import LoadImage, VisTable

root_dir = Path(__file__).resolve().parent
model_dir = os.path.join(root_dir, "models")
logger = get_logger("rapid_table_torch")
default_config = os.path.join(root_dir, "config.yaml")
ROOT_URL = "https://www.modelscope.cn/studio/jockerK/TableRec/resolve/master/models/table_rec/unitable/"
KEY_TO_MODEL_URL = {
    "unitable": {
        "encoder": f"{ROOT_URL}/encoder.pth",
        "decoder": f"{ROOT_URL}/decoder.pth",
        "vocab": f"{ROOT_URL}/vocab.json",
    }
}


class RapidTable:
    def __init__(self, encoder_path: str = None, decoder_path: str = None, vocab_path: str = None,
                 model_type: str = "unitable",
                 device: str = "cpu"):
        self.model_type = model_type
        self.load_img = LoadImage()
        encoder_path = self.get_model_path(model_type, "encoder", encoder_path)
        decoder_path = self.get_model_path(model_type, "decoder", decoder_path)
        vocab_path = self.get_model_path(model_type, "vocab", vocab_path)
        self.table_structure = TableStructurer(encoder_path, decoder_path, vocab_path, device)
        self.table_matcher = TableMatch()
        try:
            self.ocr_engine = importlib.import_module("rapidocr_onnxruntime").RapidOCR()
        except ModuleNotFoundError:
            self.ocr_engine = None

    def __call__(
            self,
            img_content: Union[str, np.ndarray, bytes, Path],
            ocr_result: List[Union[List[List[float]], str, str]] = None
    ):
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
        pred_html = self.table_matcher(pred_structures, pred_bboxes, dt_boxes, rec_res)
        logic_points = self.table_matcher.decode_logic_points(pred_structures)
        elapse = time.time() - s
        return pred_html, pred_bboxes, logic_points, elapse

    def get_boxes_recs(
            self, ocr_result: List[Union[List[List[float]], str, str]], h: int, w: int
    ):
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

    @staticmethod
    def get_model_path(model_type: str, sub_file_type: str, path: Union[str, Path, None]) -> str:
        if path is not None:
            return path

        model_url = KEY_TO_MODEL_URL.get(model_type, {}).get(sub_file_type, None)
        if model_url:
            model_path = DownloadModel.download(model_url)
            return model_path

        logger.info(
            "model url is None, using the default download model %s", path
        )
        return path
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
        "-d", "--device", type=str, default="cpu", help="device to use"
    )
    args = parser.parse_args()

    try:
        ocr_engine = importlib.import_module("rapidocr_onnxruntime").RapidOCR()
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Please install the rapidocr_onnxruntime by pip install rapidocr_onnxruntime."
        ) from exc

    rapid_table = RapidTable(device=args.device)

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