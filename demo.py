# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from rapidocr import RapidOCR

from rapid_table import ModelType, RapidTable, RapidTableInput

ocr_engine = RapidOCR()

input_args = RapidTableInput(model_type=ModelType.UNITABLE)
table_engine = RapidTable(input_args)

# img_path = "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg"

# # 使用单字识别
# ori_ocr_res = ocr_engine(img_path, return_word_box=True)
# ocr_results = [
#     [word_result[0][2], word_result[0][0], word_result[0][1]]
#     for word_result in ori_ocr_res.word_results
# ]
# ocr_results = list(zip(*ocr_results))

img_list = list(Path("images").iterdir())
results = table_engine(img_list, batch_size=3)
results.vis(save_dir="outputs", save_name="vis", indexes=(0, 1, 3))
print("ok")
