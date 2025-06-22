# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import shlex
import sys
from pathlib import Path

import pytest
from rapidocr import RapidOCR

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))
from rapid_table import EngineType, ModelType, RapidTable, RapidTableInput
from rapid_table.main import main

ocr_engine = RapidOCR()
table_engine = RapidTable()

test_file_dir = cur_dir / "test_files"
img_path = str(test_file_dir / "table.jpg")


@pytest.mark.parametrize(
    "command, expected_output",
    [(f"{img_path} --model_type slanet_plus", 1274)],
)
def test_main(capsys, command, expected_output):
    main(shlex.split(command))
    output = capsys.readouterr().out.rstrip()
    assert len(output) == expected_output


@pytest.mark.parametrize(
    "model_type,engine_type",
    [
        (ModelType.SLANETPLUS, EngineType.ONNXRUNTIME),
        (ModelType.UNITABLE, EngineType.TORCH),
    ],
)
def test_ocr_input(model_type, engine_type):
    ocr_res = ocr_engine(img_path)
    ocr_result = list(zip(ocr_res.boxes, ocr_res.txts, ocr_res.scores))

    input_args = RapidTableInput(model_type=model_type, engine_type=engine_type)
    table_engine = RapidTable(input_args)
    table_results = table_engine(img_path, ocr_result)
    assert table_results.pred_html.count("<tr>") == 16


@pytest.mark.parametrize(
    "model_type,engine_type",
    [
        (ModelType.SLANETPLUS, EngineType.ONNXRUNTIME),
        (ModelType.UNITABLE, EngineType.TORCH),
    ],
)
def test_input_ocr_none(model_type, engine_type):
    input_args = RapidTableInput(model_type=model_type, engine_type=engine_type)
    table_engine = RapidTable(input_args)
    table_results = table_engine(img_path)
    assert table_results.pred_html.count("<tr>") == 16
    assert len(table_results.cell_bboxes) == len(table_results.logic_points)
