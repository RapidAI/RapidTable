# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .model_processor.main import ModelProcessor
from .table_matcher import TableMatch
from .utils import (
    InputType,
    LoadImage,
    Logger,
    ModelType,
    RapidTableInput,
    RapidTableOutput,
    format_ocr_results,
    import_package,
    is_url,
)

logger = Logger(logger_name=__name__).get_log()
root_dir = Path(__file__).resolve().parent


class RapidTable:
    def __init__(self, cfg: Optional[RapidTableInput] = None):
        if cfg is None:
            cfg = RapidTableInput()

        if not cfg.model_dir_or_path and cfg.model_type is not None:
            cfg.model_dir_or_path = ModelProcessor.get_model_path(cfg.model_type)

        self.cfg = cfg
        self.table_structure = self._init_table_structer()

        self.ocr_engine = None
        if cfg.use_ocr:
            self.ocr_engine = self._init_ocr_engine(self.cfg.ocr_params)

        self.table_matcher = TableMatch()
        self.load_img = LoadImage()

    def _init_ocr_engine(self, params: Dict[Any, Any]):
        rapidocr_ = import_package("rapidocr")
        if rapidocr_ is None:
            logger.warning("rapidocr package is not installed, only table rec")
            return None

        if not params:
            return rapidocr_.RapidOCR()
        return rapidocr_.RapidOCR(params=params)

    def _init_table_structer(self):
        if self.cfg.model_type == ModelType.UNITABLE:
            from .table_structure.unitable import UniTableStructure

            return UniTableStructure(asdict(self.cfg))

        from .table_structure.pp_structure import PPTableStructurer

        return PPTableStructurer(asdict(self.cfg))

    def __call__(
        self,
        img_content: Union[List[InputType], InputType],
        ocr_results: Optional[Tuple[np.ndarray, Tuple[str], Tuple[float]]] = None,
        batch_size: int = 1,
    ) -> RapidTableOutput:
        s = time.perf_counter()

        imgs = self._load_imgs(img_content)

        dt_boxes, rec_res = self.get_ocr_results(imgs, ocr_results)
        pred_structures, cell_bboxes, logic_points = self.get_table_rec_results(imgs)

        pred_html = self.get_table_matcher(
            pred_structures, cell_bboxes, dt_boxes, rec_res
        )

        elapse = time.perf_counter() - s
        return RapidTableOutput(img, pred_html, cell_bboxes, logic_points, elapse)

    def _load_imgs(
        self, img_content: Union[List[InputType], InputType]
    ) -> List[np.ndarray]:
        img_contents = (
            [img_content] if isinstance(img_content, InputType) else img_content
        )
        return [self.load_img(img) for img in img_contents]

    def _batch_process(
        self,
        img_contents: List[Union[str, np.ndarray, bytes, Path]],
        ocr_results: Optional[List] = None,
        batch_size: int = 4,
    ) -> List[RapidTableOutput]:
        s = time.perf_counter()

        images = []
        for img_content in img_contents:
            img = self.load_img(img_content)
            images.append(img)

        batch_dt_boxes = []
        batch_rec_res = []

        for i, img in enumerate(images):
            img_h, img_w = img.shape[:2]
            dt_boxes, rec_res = format_ocr_results(ocr_results[i], img_h, img_w)
            batch_dt_boxes.append(dt_boxes)
            batch_rec_res.append(rec_res)

        # 批量表格结构识别
        batch_results = self.table_structure(images, batch_size)

        output_results = []
        for i, (img, (pred_structures, cell_bboxes, _)) in enumerate(
            zip(images, batch_results)
        ):
            logic_points = self.table_matcher.decode_logic_points(pred_structures)
            pred_html = self.get_table_matcher(
                pred_structures, cell_bboxes, batch_dt_boxes[i], batch_rec_res[i]
            )
            result = RapidTableOutput(img, pred_html, cell_bboxes, logic_points, 0)
            output_results.append(result)

        total_elapse = time.perf_counter() - s
        for result in output_results:
            result.elapse = total_elapse / len(output_results)

        return output_results

    def get_ocr_results(
        self,
        imgs: List[np.ndarray],
        ocr_results: Optional[List[Tuple[np.ndarray, Tuple[str], Tuple[float]]]] = None,
    ) -> Any:
        if not self.cfg.use_ocr:
            return None, None

        batch_dt_boxes, batch_rec_res = [], []

        if ocr_results is not None:
            for img, ocr_result in zip(imgs, ocr_results):
                img_h, img_w = img.shape[:2]
                dt_boxes, rec_res = format_ocr_results(ocr_result, img_h, img_w)
                batch_dt_boxes.append(dt_boxes)
                batch_rec_res.append(rec_res)
            return batch_dt_boxes, batch_rec_res

        for img in imgs:
            ori_ocr_res = self.ocr_engine(img)
            if ori_ocr_res.boxes is None:
                logger.warning("OCR Result is empty")
                batch_dt_boxes.append(None)
                batch_rec_res.append(None)

            img_h, img_w = img.shape[:2]
            ocr_result = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]
            dt_boxes, rec_res = format_ocr_results(ocr_result, img_h, img_w)
            batch_dt_boxes.append(dt_boxes)
            batch_rec_res.append(rec_res)

        return batch_dt_boxes, batch_rec_res

    def get_table_rec_results(self, imgs: List[np.ndarray]):
        pred_structures, cell_bboxes, _ = self.table_structure(imgs)
        logic_points = self.table_matcher.decode_logic_points(pred_structures)
        return pred_structures, cell_bboxes, logic_points

    def get_table_matcher(self, pred_structures, cell_bboxes, dt_boxes, rec_res):
        if dt_boxes is None and rec_res is None:
            return None

        return self.table_matcher(pred_structures, cell_bboxes, dt_boxes, rec_res)


def parse_args(arg_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="the image path or URL of the table")
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
        save_dir = Path(".") if is_url(img_path) else Path(img_path).resolve().parent
        table_results.vis(save_dir, save_name=Path(img_path).stem)


if __name__ == "__main__":
    main()
