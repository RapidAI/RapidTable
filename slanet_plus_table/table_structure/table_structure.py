import time
import numpy as np
from .utils import TablePredictor, TablePreprocess, TableLabelDecode


# class SLANetPlus:
#     def __init__(self, model_dir, model_prefix="inference"):
#         self.preprocess_op = TablePreprocess()
#
#         self.mean=[0.485, 0.456, 0.406]
#         self.std=[0.229, 0.224, 0.225]
#         self.target_img_size = [488, 488]
#         self.scale=1 / 255
#         self.order="hwc"
#         self.img_loader = LoadImage()
#         self.target_size = 488
#         self.pad_color = 0
#         self.predictor = TablePredictor(model_dir, model_prefix)
#         dict_character=['sos', '<thead>', '</thead>', '<tbody>', '</tbody>', '<tr>', '</tr>', '<td', '>', '</td>', ' colspan="2"', ' colspan="3"', ' colspan="4"', ' colspan="5"', ' colspan="6"', ' colspan="7"', ' colspan="8"', ' colspan="9"', ' colspan="10"', ' colspan="11"', ' colspan="12"', ' colspan="13"', ' colspan="14"', ' colspan="15"', ' colspan="16"', ' colspan="17"', ' colspan="18"', ' colspan="19"', ' colspan="20"', ' rowspan="2"', ' rowspan="3"', ' rowspan="4"', ' rowspan="5"', ' rowspan="6"', ' rowspan="7"', ' rowspan="8"', ' rowspan="9"', ' rowspan="10"', ' rowspan="11"', ' rowspan="12"', ' rowspan="13"', ' rowspan="14"', ' rowspan="15"', ' rowspan="16"', ' rowspan="17"', ' rowspan="18"', ' rowspan="19"', ' rowspan="20"', '<td></td>', 'eos']
#         self.beg_str = "sos"
#         self.end_str = "eos"
#         self.dict = {}
#         self.table_matcher = TableMatch()
#         for i, char in enumerate(dict_character):
#             self.dict[char] = i
#         self.character = dict_character
#         self.td_token = ["<td>", "<td", "<td></td>"]
#
#     def call(self, img):
#         starttime = time.time()
#         data = {"image": img}
#         data = self.preprocess_op(data)
#         img = data[0]
#         if img is None:
#             return None, 0
#         img = np.expand_dims(img, axis=0)
#         img = img.copy()
#     def __call__(self, img, ocr_result):
#         img = self.img_loader(img)
#         h, w = img.shape[:2]
#         n_img, h_resize, w_resize = self.resize(img)
#         n_img = self.normalize(n_img)
#         n_img = self.pad(n_img)
#         n_img = n_img.transpose((2, 0, 1))
#         n_img = np.expand_dims(n_img, axis=0)
#         start = time.time()
#         batch_output = self.predictor(n_img)
#         elapse_time = time.time() - start
#         ori_img_size = [[w, h]]
#         output = self.decode(batch_output, ori_img_size)[0]
#         corners = np.stack(output['bbox'], axis=0)
#         dt_boxes, rec_res = get_boxes_recs(ocr_result, h, w)
#         pred_html = self.table_matcher(output['structure'], convert_corners_to_bounding_boxes(corners), dt_boxes, rec_res)
#         return pred_html,output['bbox'], elapse_time
#     def resize(self, img):
#         h, w = img.shape[:2]
#         scale = self.target_size / max(h, w)
#         h_resize = round(h * scale)
#         w_resize = round(w * scale)
#         resized_img = cv2.resize(img, (w_resize, h_resize), interpolation=cv2.INTER_LINEAR)
#         return resized_img, h_resize, w_resize
#     def pad(self, img):
#         h, w = img.shape[:2]
#         tw, th = self.target_img_size
#         ph = th - h
#         pw = tw - w
#         pad = (0, ph, 0, pw)
#         chns = 1 if img.ndim == 2 else img.shape[2]
#         im = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=(self.pad_color,) * chns)
#         return im
#     def normalize(self, img):
#         img = img.astype("float32", copy=False)
#         img *= self.scale
#         img -= self.mean
#         img /= self.std
#         return img
#
#
#     def decode(self, pred, ori_img_size):
#         bbox_preds, structure_probs = [], []
#         for bbox_pred, stru_prob in pred:
#             bbox_preds.append(bbox_pred)
#             structure_probs.append(stru_prob)
#         bbox_preds = np.array(bbox_preds)
#         structure_probs = np.array(structure_probs)
#
#         bbox_list, structure_str_list, structure_score = self.decode_single(
#             structure_probs, bbox_preds, [self.target_img_size], ori_img_size
#         )
#         structure_str_list = [
#             (
#                     ["<html>", "<body>", "<table>"]
#                     + structure
#                     + ["</table>", "</body>", "</html>"]
#             )
#             for structure in structure_str_list
#         ]
#         return [
#             {"bbox": bbox, "structure": structure, "structure_score": structure_score}
#             for bbox, structure in zip(bbox_list, structure_str_list)
#         ]
#
#
#     def decode_single(self, structure_probs, bbox_preds, padding_size, ori_img_size):
#         """convert text-label into text-index."""
#         ignored_tokens = [self.beg_str, self.end_str]
#         end_idx = self.dict[self.end_str]
#
#         structure_idx = structure_probs.argmax(axis=2)
#         structure_probs = structure_probs.max(axis=2)
#
#         structure_batch_list = []
#         bbox_batch_list = []
#         batch_size = len(structure_idx)
#         for batch_idx in range(batch_size):
#             structure_list = []
#             bbox_list = []
#             score_list = []
#             for idx in range(len(structure_idx[batch_idx])):
#                 char_idx = int(structure_idx[batch_idx][idx])
#                 if idx > 0 and char_idx == end_idx:
#                     break
#                 if char_idx in ignored_tokens:
#                     continue
#                 text = self.character[char_idx]
#                 if text in self.td_token:
#                     bbox = bbox_preds[batch_idx, idx]
#                     bbox = self._bbox_decode(
#                         bbox, padding_size[batch_idx], ori_img_size[batch_idx]
#                     )
#                     bbox_list.append(bbox.astype(int))
#                 structure_list.append(text)
#                 score_list.append(structure_probs[batch_idx, idx])
#             structure_batch_list.append(structure_list)
#             structure_score = np.mean(score_list)
#             bbox_batch_list.append(bbox_list)
#
#         return bbox_batch_list, structure_batch_list, structure_score
#
#     def _bbox_decode(self, bbox, padding_shape, ori_shape):
#
#         pad_w, pad_h = padding_shape
#         w, h = ori_shape
#         ratio_w = pad_w / w
#         ratio_h = pad_h / h
#         ratio = min(ratio_w, ratio_h)
#
#         bbox[0::2] *= pad_w
#         bbox[1::2] *= pad_h
#         bbox[0::2] /= ratio
#         bbox[1::2] /= ratio
#
#         return bbox


class TableStructurer:
    def __init__(self, model_path: str):
        self.preprocess_op = TablePreprocess()
        self.predictor = TablePredictor(model_path)
        self.character = ['<thead>', '</thead>', '<tbody>', '</tbody>', '<tr>', '</tr>', '<td', '>', '</td>', ' colspan="2"', ' colspan="3"', ' colspan="4"', ' colspan="5"', ' colspan="6"', ' colspan="7"', ' colspan="8"', ' colspan="9"', ' colspan="10"', ' colspan="11"', ' colspan="12"', ' colspan="13"', ' colspan="14"', ' colspan="15"', ' colspan="16"', ' colspan="17"', ' colspan="18"', ' colspan="19"', ' colspan="20"', ' rowspan="2"', ' rowspan="3"', ' rowspan="4"', ' rowspan="5"', ' rowspan="6"', ' rowspan="7"', ' rowspan="8"', ' rowspan="9"', ' rowspan="10"', ' rowspan="11"', ' rowspan="12"', ' rowspan="13"', ' rowspan="14"', ' rowspan="15"', ' rowspan="16"', ' rowspan="17"', ' rowspan="18"', ' rowspan="19"', ' rowspan="20"', '<td></td>']
        self.postprocess_op = TableLabelDecode(self.character)

    def __call__(self, img):
        start_time = time.time()
        data = {"image": img}
        h, w = img.shape[:2]
        ori_img_size = [[w, h]]
        data = self.preprocess_op(data)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()
        cur_img_size = [[488, 488]]
        outputs = self.predictor(img)
        output = self.postprocess_op(outputs, cur_img_size, ori_img_size)[0]
        elapse = time.time() - start_time
        return output["structure"], np.stack(output["bbox"]), elapse
