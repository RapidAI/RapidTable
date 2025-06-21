# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapidocr import RapidOCR

from rapid_table import ModelType, RapidTable, RapidTableInput

ocr_engine = RapidOCR()

input_args = RapidTableInput(model_type=ModelType.UNITABLE)
table_engine = RapidTable(input_args)

img_path = "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg"
rapid_ocr_output = ocr_engine(img_path)
ocr_result = list(
    zip(rapid_ocr_output.boxes, rapid_ocr_output.txts, rapid_ocr_output.scores)
)
results = table_engine(img_path, ocr_result)
results.vis(save_dir="outputs", save_name="1")
