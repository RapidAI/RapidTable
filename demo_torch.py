# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

from rapid_table import RapidTable, VisTable
from rapid_table.table_structure.utils import trans_char_ocr_res

ocr_engine = RapidOCR()
table_engine = RapidTable(use_cuda=True, device="cuda:0", model_type="unitable")
viser = VisTable()
img_path = "tests/test_files/table.jpg"
# OCR
ocr_result, _ = ocr_engine(img_path, return_word_box=True)
ocr_result = trans_char_ocr_res(ocr_result)
boxes, txts, scores = list(zip(*ocr_result))
# Save
save_dir = Path("outputs")
save_dir.mkdir(parents=True, exist_ok=True)

save_html_path = save_dir / f"{Path(img_path).stem}.html"
save_drawed_path = save_dir / f"{Path(img_path).stem}_table_vis{Path(img_path).suffix}"

table_html_str, table_cell_bboxes, elapse = table_engine(img_path, ocr_result)
viser(img_path, table_html_str, save_html_path, table_cell_bboxes, save_drawed_path)
# 返回逻辑坐标
# table_html_str, table_cell_bboxes, logic_points, elapse = table_engine(img_path, ocr_result, return_logic_points=True)
# save_logic_path = save_dir / f"vis_logic_{Path(img_path).name}"
# vis_imged = viser(img_path, table_html_str, save_html_path, table_cell_bboxes, save_drawed_path, logic_points,
#                   save_logic_path)
print(f"elapse:{elapse}")
