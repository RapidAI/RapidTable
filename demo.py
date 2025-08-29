# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from rapid_table import ModelType, RapidTable, RapidTableInput

# input_args = RapidTableInput(
#     model_type=ModelType.UNITABLE,
#     engine_cfg={"use_cuda": True, "gpu_id": 1},
# )
input_args = RapidTableInput(model_type=ModelType.PPSTRUCTURE_ZH)
table_engine = RapidTable(input_args)

img_list = list(Path("images").iterdir())
results = table_engine(img_list, batch_size=3)
results.vis(save_dir="outputs", save_name="vis", indexes=(0, 1, 2))
