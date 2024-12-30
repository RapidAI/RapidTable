# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

from rapidocr_onnxruntime import RapidOCR

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from rapid_table_torch import RapidTable

ocr_engine = RapidOCR()
table_engine = RapidTable()

test_file_dir = cur_dir / "test_files"
img_path = str(test_file_dir / "table.jpg")


def test_ocr_input():
    ocr_res, _ = ocr_engine(img_path)
    table_html_str, table_cell_bboxes, logic_points, elapse = table_engine(img_path, ocr_res)
    assert table_html_str.count("<tr>") == 16


def test_input_ocr_none():
    table_html_str, table_cell_bboxes, logic_points, elapse = table_engine(img_path)
    assert table_html_str.count("<tr>") == 16

def test_logic_points_out():
    table_html_str, table_cell_bboxes, logic_points, elapse = table_engine(img_path, return_logic_points=True)
    assert len(table_cell_bboxes) == len(logic_points)