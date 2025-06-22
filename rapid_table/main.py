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
    get_boxes_recs,
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

        self.ocr_engine = None
        if cfg.use_ocr:
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
        ocr_results: Optional[Tuple[np.ndarray, Tuple[str], Tuple[float]]] = None,
    ) -> RapidTableOutput:
        s = time.perf_counter()

        img = self.load_img(img_content)

        dt_boxes, rec_res = self.get_ocr_results(img, ocr_results)
        pred_structures, cell_bboxes, logic_points = self.get_table_rec_results(img)
        pred_html = self.get_table_matcher(
            pred_structures, cell_bboxes, dt_boxes, rec_res
        )

        elapse = time.perf_counter() - s
        return RapidTableOutput(img, pred_html, cell_bboxes, logic_points, elapse)

    def get_ocr_results(
        self, img: np.ndarray, ocr_results: Tuple[np.ndarray, Tuple[str], Tuple[float]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if ocr_results is not None:
            return get_boxes_recs(ocr_results, img.shape[:2])

        if not self.cfg.use_ocr:
            return None, None

        ori_ocr_res = self.ocr_engine(img)
        if ori_ocr_res.boxes is None:
            logger.warning("OCR Result is empty")
            return None, None

        ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]
        return get_boxes_recs(ocr_results, img.shape[:2])

    def get_table_rec_results(self, img: np.ndarray):
        pred_structures, cell_bboxes, _ = self.table_structure(img)
        logic_points = self.table_matcher.decode_logic_points(pred_structures)
        return pred_structures, cell_bboxes, logic_points

    def get_table_matcher(self, pred_structures, cell_bboxes, dt_boxes, rec_res):
        if dt_boxes is None and rec_res is None:
            return None

        return self.table_matcher(pred_structures, cell_bboxes, dt_boxes, rec_res)


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

    ori_ocr_res = table_engine.ocr_engine(img_path)
    ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]
    table_results = table_engine(img_path, ocr_results=ocr_results)
    print(table_results.pred_html)

    if args.vis:
        save_dir = img_path.resolve().parent
        table_results.vis(save_dir, save_name=img_path.stem)


if __name__ == "__main__":
    main()
