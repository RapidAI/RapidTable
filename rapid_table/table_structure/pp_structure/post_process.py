# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import List, Optional

import numpy as np


class TableLabelDecode:
    def __init__(self, dict_character, merge_no_span_structure=True):
        if merge_no_span_structure:
            if "<td></td>" not in dict_character:
                dict_character.append("<td></td>")
            if "<td>" in dict_character:
                dict_character.remove("<td>")

        dict_character = self.add_special_char(dict_character)
        self.char_to_index = {}
        for i, char in enumerate(dict_character):
            self.char_to_index[char] = i

        self.character = dict_character
        self.td_token = ["<td>", "<td", "<td></td>"]

    def __call__(
        self,
        bbox_preds: np.ndarray,
        structure_probs: np.ndarray,
        batch: Optional[List[np.ndarray]],
    ):
        shape_list = batch[-1]
        result = self.decode(structure_probs, bbox_preds, shape_list)
        if len(batch) == 1:
            # only contains shape
            return result

        label_decode_result = self.decode_label(batch)
        return result, label_decode_result

    def decode(self, structure_probs, bbox_preds, shape_list):
        """convert text-label into text-index."""
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.char_to_index[self.end_str]

        structure_idx = structure_probs.argmax(axis=2)
        structure_probs = structure_probs.max(axis=2)

        structure_batch_list = []
        bbox_batch_list = []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list, bbox_list, score_list = [], [], []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break

                if char_idx in ignored_tokens:
                    continue

                text = self.character[char_idx]
                if text in self.td_token:
                    bbox = bbox_preds[batch_idx, idx]
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)

                structure_list.append(text)
                score_list.append(structure_probs[batch_idx, idx])
            structure_batch_list.append([structure_list, np.mean(score_list)])
            bbox_batch_list.append(np.array(bbox_list))

        return {
            "bbox_batch_list": bbox_batch_list,
            "structure_batch_list": structure_batch_list,
        }

    def decode_label(self, batch):
        """convert text-label into text-index."""
        structure_idx = batch[1]
        gt_bbox_list = batch[2]
        shape_list = batch[-1]
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.char_to_index[self.end_str]

        structure_batch_list = []
        bbox_batch_list = []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list = []
            bbox_list = []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break

                if char_idx in ignored_tokens:
                    continue

                structure_list.append(self.character[char_idx])

                bbox = gt_bbox_list[batch_idx][idx]
                if bbox.sum() != 0:
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)

            structure_batch_list.append(structure_list)
            bbox_batch_list.append(bbox_list)
        result = {
            "bbox_batch_list": bbox_batch_list,
            "structure_batch_list": structure_batch_list,
        }
        return result

    def _bbox_decode(self, bbox, shape):
        h, w = shape[:2]
        bbox[0::2] *= w
        bbox[1::2] *= h
        return bbox

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            return np.array(self.char_to_index[self.beg_str])

        if beg_or_end == "end":
            return np.array(self.char_to_index[self.end_str])

        raise TypeError(f"unsupport type {beg_or_end} in get_beg_end_flag_idx")

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character
