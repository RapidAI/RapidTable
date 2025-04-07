# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from rapidocr import RapidOCR, VisRes

from rapid_table import RapidTable, RapidTableInput, VisTable

if __name__ == "__main__":
    # Init
    ocr_engine = RapidOCR()
    vis_ocr = VisRes()

    input_args = RapidTableInput(model_type="unitable")
    table_engine = RapidTable(input_args)
    viser = VisTable()

    img_path = "tests/test_files/table.jpg"

    # OCR
    rapid_ocr_output = ocr_engine(img_path)
    ocr_result = list(
        zip(rapid_ocr_output.boxes, rapid_ocr_output.txts, rapid_ocr_output.scores)
    )
    table_results = table_engine(img_path, ocr_result)
    # 使用单字识别
    # word_results = rapid_ocr_output.word_results
    # ocr_result = [
    #     [word_result[2], word_result[0], word_result[1]] for word_result in word_results
    # ]
    # table_results = table_engine(img_path, ocr_result)

    table_html_str, table_cell_bboxes = (
        table_results.pred_html,
        table_results.cell_bboxes,
    )
    # Save
    save_dir = Path("outputs")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_html_path = save_dir / f"{Path(img_path).stem}.html"
    save_drawed_path = (
        save_dir / f"{Path(img_path).stem}_table_vis{Path(img_path).suffix}"
    )
    save_logic_points_path = (
        save_dir / f"{Path(img_path).stem}_table_col_row_vis{Path(img_path).suffix}"
    )

    # Visualize table rec result
    vis_imged = viser(
        img_path,
        table_results,
        save_html_path,
        save_drawed_path,
        save_logic_points_path,
    )

    print(f"The results has been saved {save_dir}")
