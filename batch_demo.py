# -*- encoding: utf-8 -*-
# @Author: deeperrrr
# @Contact: 3545615231@qq.com
import cv2
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm

from rapidocr import EngineType, RapidOCR
from rapid_table import ModelType, RapidTable, RapidTableInput

ocr_engine = RapidOCR(
    params={
        "Det.engine_type": EngineType.TORCH,
        "Cls.engine_type": EngineType.TORCH,
        "Rec.engine_type": EngineType.TORCH,
    }
)
img_dir_path = "/data/images"  # 图片文件夹
ocr_results = []
batch_size = 4

# input_args = RapidTableInput(model_type=ModelType.UNITABLE)
input_args = RapidTableInput(model_type=ModelType.SLANETPLUS)
table_engine = RapidTable(input_args)


def load_images_original_size(img_dir: str) -> List[np.ndarray]:
    img_dir = Path(img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"目录不存在: {img_dir}")

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(list(img_dir.glob(ext)))

    images = []
    for img_path in tqdm(image_paths, desc="加载图像"):
        img = cv2.imread(str(img_path))
        images.append(img)
    return images


def dynamic_batch_process(table_engine: RapidTable, images: List[np.ndarray], ocr_results: List[List],
                          batch_size: int = 1):
    all_results = []
    for i in tqdm(range(0, len(images), batch_size), desc=f"表格批量推理, batch_size={batch_size}"):
        batch_imgs = images[i:i + batch_size]
        batch_ocrs = ocr_results[i:i + batch_size]
        results = table_engine(batch_imgs, batch_ocrs, batch_size)
        all_results.extend(results)
    return all_results


images = load_images_original_size(img_dir_path)
for img in tqdm(images, desc="OCR处理"):
    ori_ocr_res = ocr_engine(img)
    ocr_result = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]
    ocr_results.append(ocr_result)

# 批量表格结构识别
results = dynamic_batch_process(table_engine, images, ocr_results, batch_size)  # batch_size默认4

for i, result in enumerate(results):
    result.vis(save_dir="outputs", save_name=f"vis_{i}")