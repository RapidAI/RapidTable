import copy
import importlib
import time
from pathlib import Path
from typing import Optional, Union, List, Tuple

import cv2
import numpy as np

from slanet_plus_table.table_matcher import TableMatch
from slanet_plus_table.table_structure import TableStructurer
from slanet_plus_table.utils import LoadImage, VisTable

root_dir = Path(__file__).resolve().parent


class SLANetPlus:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = str(
                root_dir / "models"
            )

        self.load_img = LoadImage()
        self.table_structure = TableStructurer(model_path)
        self.table_matcher = TableMatch()

        try:
            self.ocr_engine = importlib.import_module("rapidocr_onnxruntime").RapidOCR()
        except ModuleNotFoundError:
            self.ocr_engine = None

    def __call__(
            self,
            img_content: Union[str, np.ndarray, bytes, Path],
            ocr_result: List[Union[List[List[float]], str, str]] = None,
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
        pred_html = self.table_matcher(pred_structures, pred_bboxes, dt_boxes, rec_res)

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


if __name__ == '__main__':
    slanet_table = SLANetPlus()
    img_path = "D:\pythonProjects\TableStructureRec\outputs\\benchmark\\border_left_7267_OEJGHZF525Q011X2ZC34.jpg"
    img = cv2.imread(img_path)
    try:
        ocr_engine = importlib.import_module("rapidocr_onnxruntime").RapidOCR()
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Please install the rapidocr_onnxruntime by pip install rapidocr_onnxruntime."
        ) from exc
    ocr_result, _ = ocr_engine(img)
    table_html_str, table_cell_bboxes, elapse = slanet_table(img, ocr_result)

    viser = VisTable()

    img_path = Path(img_path)

    save_dir = "outputs"
    save_html_path = f"{save_dir}/{Path(img_path).stem}.html"
    save_drawed_path = f"{save_dir}/vis_{Path(img_path).name}"
    viser(
        img_path,
        table_html_str,
        save_html_path,
        table_cell_bboxes,
        save_drawed_path,
    )
