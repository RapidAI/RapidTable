# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import List


def get_struct_str(structure_str_list: List[str]) -> List[str]:
    structure_str_list = (
        ["<html>", "<body>", "<table>"]
        + structure_str_list
        + ["</table>", "</body>", "</html>"]
    )
    return structure_str_list
