# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .model_processor.main import ModelProcessor
from .table_matcher import TableMatch
from .utils import (
    LoadImage,
    Logger,
    ModelType,
    RapidTableInput,
    RapidTableOutput,
    import_package,
)

logger = Logger(logger_name=__name__).get_log()
root_dir = Path(__file__).resolve().parent


class RapidTable:
    def __init__(self, cfg: Optional[RapidTableInput] = None):
        if cfg is None:
            cfg = RapidTableInput()

        if not cfg.model_dir_or_path:
            cfg.model_dir_or_path = ModelProcessor.get_model_path(cfg.model_type)

        self.cfg = cfg
        self.table_structure = self._init_table_structer()
        self.ocr_engine = self._init_ocr_engine()
        self.table_matcher = TableMatch()
        self.load_img = LoadImage()

    def _init_ocr_engine(self):
        try:
            return import_package("rapidocr").RapidOCR()
        except ModuleNotFoundError:
            logger.warning("rapidocr package is not installed, only table rec")
            return None

    def _init_table_structer(self):
        if self.cfg.model_type == ModelType.UNITABLE:
            from .table_structure.unitable import UniTableStructure

            return UniTableStructure(asdict(self.cfg))

        from .table_structure.pp_structure import PPTableStructurer

        return PPTableStructurer(asdict(self.cfg))

    def __call__(
        self,
        img_content: Union[str, np.ndarray, bytes, Path],
        ocr_result: Optional[List[Union[List[List[float]], str, str]]] = None,
    ) -> RapidTableOutput:
        if self.ocr_engine is None and ocr_result is None:
            raise ValueError(
                "One of two conditions must be met: ocr_result is not empty, or rapidocr is installed."
            )

        img = self.load_img(img_content)

        s = time.perf_counter()
        h, w = img.shape[:2]

        if ocr_result is None:
            ocr_result = self.ocr_engine(img)
            ocr_result = list(
                zip(
                    ocr_result.boxes,
                    ocr_result.txts,
                    ocr_result.scores,
                )
            )
        dt_boxes, rec_res = self.get_boxes_recs(ocr_result, h, w)

        pred_structures, cell_bboxes, _ = self.table_structure(img)

        # 适配slanet-plus模型输出的box缩放还原
        if self.cfg.model_type == ModelType.SLANETPLUS:
            cell_bboxes = self.adapt_slanet_plus(img, cell_bboxes)

        pred_html = self.table_matcher(pred_structures, cell_bboxes, dt_boxes, rec_res)

        # 过滤掉占位的bbox
        mask = ~np.all(cell_bboxes == 0, axis=1)
        cell_bboxes = cell_bboxes[mask]

        logic_points = self.table_matcher.decode_logic_points(pred_structures)
        elapse = time.perf_counter() - s
        return RapidTableOutput(img, pred_html, cell_bboxes, logic_points, elapse)

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

    def adapt_slanet_plus(self, img: np.ndarray, cell_bboxes: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        resized = 488
        ratio = min(resized / h, resized / w)
        w_ratio = resized / (w * ratio)
        h_ratio = resized / (h * ratio)
        cell_bboxes[:, 0::2] *= w_ratio
        cell_bboxes[:, 1::2] *= h_ratio
        return cell_bboxes


def parse_args(arg_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=Path, help="Path to image for layout.")
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default=ModelType.SLANETPLUS.value,
        choices=[v.value for v in ModelType],
        help="Supported table rec models",
    )
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        default=False,
        help="Wheter to visualize the layout results.",
    )
    args = parser.parse_args(arg_list)
    return args


def main(arg_list: Optional[List[str]] = None):
    args = parse_args(arg_list)
    img_path = args.img_path

    input_args = RapidTableInput(model_type=ModelType(args.model_type))
    table_engine = RapidTable(input_args)

    if table_engine.ocr_engine is None:
        raise ValueError("ocr engine is None")

    rapid_ocr_output = table_engine.ocr_engine(img_path)
    ocr_result = list(
        zip(rapid_ocr_output.boxes, rapid_ocr_output.txts, rapid_ocr_output.scores)
    )
    table_results = table_engine(img_path, ocr_result)
    print(table_results.pred_html)

    if args.vis:
        save_dir = img_path.resolve().parent
        table_results.vis(save_dir, save_name=img_path.stem)


if __name__ == "__main__":
    main()
