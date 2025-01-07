# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import pytest
from rapidocr_onnxruntime import RapidOCR

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from rapid_table import RapidTable, RapidTableInput

ocr_engine = RapidOCR()

input_args = RapidTableInput()
table_engine = RapidTable(input_args)

test_file_dir = cur_dir / "test_files"
img_path = str(test_file_dir / "table.jpg")


@pytest.mark.parametrize("model_type", ["slanet_plus", "unitable"])
def test_ocr_input(model_type):
    ocr_res, _ = ocr_engine(img_path)

    input_args = RapidTableInput(model_type=model_type)
    table_engine = RapidTable(input_args)

    table_results = table_engine(img_path, ocr_res)
    assert table_results.pred_html.count("<tr>") == 16


@pytest.mark.parametrize("model_type", ["slanet_plus", "unitable"])
def test_input_ocr_none(model_type):
    input_args = RapidTableInput(model_type=model_type)
    table_engine = RapidTable(input_args)
    table_results = table_engine(img_path)
    assert table_results.pred_html.count("<tr>") == 16


@pytest.mark.parametrize("model_type", ["slanet_plus", "unitable"])
def test_logic_points_out(model_type):
    input_args = RapidTableInput(model_type=model_type)
    table_engine = RapidTable(input_args)
    table_results = table_engine(img_path, return_logic_points=True)
    assert len(table_results.pred_bboxes) == len(table_results.logic_points)
