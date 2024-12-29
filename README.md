<div align="center">
  <div align="center">
    <h1><b>📊 Rapid Table</b></h1>
  </div>

<a href="https://swhl-rapidstructuredemo.hf.space" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Online Demo-blue"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.13-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://pypi.org/project/rapid-table/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapid-table"></a>
<a href="https://pepy.tech/project/rapid-table"><img src="https://static.pepy.tech/personalized-badge/rapid-table?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

</div>

### 简介

RapidTable库是专门用来文档类图像的表格结构还原，表格结构模型均属于序列预测方法，结合RapidOCR，将给定图像中的表格转化对应的HTML格式。

slanet_plus是paddlex内置的SLANet升级版模型，准确率有大幅提升

unitable是来源unitable的transformer模型，精度最高，暂仅支持pytorch推理，支持gpu推理加速,训练权重来源于 [OhMyTable项目](https://github.com/Sanster/OhMyTable)

|      模型类型      |                  模型名称                  | 推理框架 |模型大小 |推理耗时(单图 60KB)|
  |:--------------:|:--------------------------------------:| :------: |:------: |:------: |
|       英文       | `en_ppstructure_mobile_v2_SLANet.onnx` |   onnxruntime   |7.3M |0.15s |
|       中文       | `ch_ppstructure_mobile_v2_SLANet.onnx` |   onnxruntime   |7.4M |0.15s |
| slanet_plus 中文 |          `slanet-plus.onnx`           |  onnxruntime    |6.8M |0.15s |
| unitable 中文 |          `unitable(encoder.pth,decoder.pth)` |  pytorch    |500M |cpu(6s) gpu-4090(1.5s)|

模型来源\
[PaddleOCR 表格识别](https://github.com/PaddlePaddle/PaddleOCR/blob/133d67f27dc8a241d6b2e30a9f047a0fb75bebbe/ppstructure/table/README_ch.md) \
[PaddleX-SlaNetPlus 表格识别](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/table_structure_recognition.md) \
[Unitable](https://github.com/poloclub/unitable?tab=readme-ov-file)

模型下载地址为：[link](https://github.com/RapidAI/RapidTable/releases/tag/assets)

### 效果展示

<div align="center">
    <img src="https://github.com/RapidAI/RapidTable/releases/download/assets/preview.gif" alt="Demo" width="100%" height="100%">
</div>

### 更新日志

<details>

#### 2024.12.30 update

- 支持Unitable模型的表格识别，使用pytorch框架

#### 2024.11.24 update

- 支持gpu推理，适配 rapidOCR 单字识别匹配,支持逻辑坐标返回及可视化

#### 2024.10.13 update

- 补充最新paddlex-SLANet-plus 模型(paddle2onnx原因暂不能支持onnx)

#### 2023-12-29 v0.1.3 update

- 优化可视化结果部分

#### 2023-12-27 v0.1.2 update

- 添加返回cell坐标框参数
- 完善可视化函数

#### 2023-07-17 v0.1.0 update

- 将`rapidocr_onnxruntime`部分从`rapid_table`中解耦合出来，给出选项是否依赖，更加灵活。

- 增加接口输入参数`ocr_result`：
    - 如果在调用函数时，事先指定了`ocr_result`参数值，则不会再走OCR。其中`ocr_result`格式需要和`rapidocr_onnxruntime`返回值一致。
    - 如果未指定`ocr_result`参数值，但是事先安装了`rapidocr_onnxruntime`库，则会自动调用该库，进行识别。
    - 如果`ocr_result`未指定，且`rapidocr_onnxruntime`未安装，则会报错。必须满足两个条件中一个。

#### 2023-07-10 v0.0.13 updata

- 更改传入表格还原中OCR的实例接口，可以传入其他OCR实例，前提要与`rapidocr_onnxruntime`接口一致

#### 2023-07-06 v0.0.12 update

- 去掉返回表格的html字符串中的`<thead></thead><tbody></tbody>`元素，便于后续统一。
- 采用Black工具优化代码

</details>

### 与[TableStructureRec](https://github.com/RapidAI/TableStructureRec)关系

TableStructureRec库是一个表格识别算法的集合库，当前有`wired_table_rec`有线表格识别算法和`lineless_table_rec`无线表格识别算法的推理包。

RapidTable是整理自PP-Structure中表格识别部分而来。由于PP-Structure较早，这个库命名就成了`rapid_table`。

总之，RapidTable和TabelStructureRec都是表格识别的仓库。大家可以都试试，哪个好用用哪个。由于每个算法都不太同，暂时不打算做统一处理。

关于表格识别算法的比较，可参见[TableStructureRec测评](https://github.com/RapidAI/TableStructureRec#指标结果)

### 安装

由于模型较小，预先将slanet-plus表格识别模型(`slanet-plus.onnx`)打包进了whl包内。

> ⚠️注意：`rapid_table>=v0.1.0`之后，不再将`rapidocr_onnxruntime`依赖强制打包到`rapid_table`中。使用前，需要自行安装`rapidocr_onnxruntime`包。

```bash
pip install rapidocr_onnxruntime
pip install rapid_table
#pip install rapid_table_torch # for unitable inference 
#pip install onnxruntime-gpu # for gpu inference
```

### 使用方式

#### python脚本运行

RapidTable类提供model_path参数，可以自行指定上述2个模型，默认是`slanet-plus.onnx`。举例如下：

```python
table_engine = RapidTable()
```

完整示例：

#### onnx版本
```python
from pathlib import Path

from rapid_table import RapidTable, VisTable
from rapidocr_onnxruntime import RapidOCR
from rapid_table.table_structure.utils import trans_char_ocr_res

table_engine = RapidTable()
# 开启onnx-gpu推理
# table_engine = RapidTable(use_cuda=True)
ocr_engine = RapidOCR()
viser = VisTable()

img_path = 'test_images/table.jpg'

ocr_result, _ = ocr_engine(img_path)
# 单字匹配
# ocr_result, _ = ocr_engine(img_path, return_word_box=True)
# ocr_result = trans_char_ocr_res(ocr_result)
table_html_str, table_cell_bboxes, elapse = table_engine(img_path, ocr_result)

save_dir = Path("./inference_results/")
save_dir.mkdir(parents=True, exist_ok=True)

save_html_path = save_dir / f"{Path(img_path).stem}.html"
save_drawed_path = save_dir / f"vis_{Path(img_path).name}"

viser(img_path, table_html_str, save_html_path, table_cell_bboxes, save_drawed_path)

# 返回逻辑坐标
# table_html_str, table_cell_bboxes, logic_points, elapse = table_engine(img_path, ocr_result, return_logic_points=True)
# save_logic_path = save_dir / f"vis_logic_{Path(img_path).name}"
# viser(img_path, table_html_str, save_html_path, table_cell_bboxes, save_drawed_path,logic_points, save_logic_path)

print(table_html_str)
```

#### torch版本
```python
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

from rapid_table_torch import RapidTable, VisTable
from rapid_table_torch.table_structure.utils import trans_char_ocr_res

if __name__ == '__main__':
# Init
ocr_engine = RapidOCR()
table_engine = RapidTable(device="cpu") # 默认使用cpu，若使用cuda，则传入device="cuda:0"
viser = VisTable()
img_path = "tests/test_files/image34.png"
# OCR,本模型检测框比较精准，配合单字匹配效果更好
ocr_result, _ = ocr_engine(img_path, return_word_box=True)
ocr_result = trans_char_ocr_res(ocr_result)
boxes, txts, scores = list(zip(*ocr_result))
# Save
save_dir = Path("outputs")
save_dir.mkdir(parents=True, exist_ok=True)

save_html_path = save_dir / f"{Path(img_path).stem}.html"
save_drawed_path = save_dir / f"{Path(img_path).stem}_table_vis{Path(img_path).suffix}"
# 返回逻辑坐标
table_html_str, table_cell_bboxes, logic_points, elapse = table_engine(img_path, ocr_result)
save_logic_path = save_dir / f"vis_logic_{Path(img_path).name}"
vis_imged = viser(img_path, table_html_str, save_html_path, table_cell_bboxes, save_drawed_path, logic_points,
                  save_logic_path)
print(f"elapse:{elapse}")
```

#### 终端运行

##### onnx:
- 用法:

  ```bash
  $ rapid_table -h
  usage: rapid_table [-h] [-v] -img IMG_PATH [-m MODEL_PATH]

  optional arguments:
  -h, --help            show this help message and exit
  -v, --vis             Whether to visualize the layout results.
  -img IMG_PATH, --img_path IMG_PATH
                        Path to image for layout.
  -m MODEL_PATH, --model_path MODEL_PATH
                        The model path used for inference.
  ```

- 示例:

  ```bash
  rapid_table -v -img test_images/table.jpg
  ```

##### pytorch:
- 用法:

  ```bash
  $ rapid_table_torch -h
  usage: rapid_table_torch [-h] [-v] -img IMG_PATH [-d DEVICE]

  optional arguments:
  -h, --help            show this help message and exit
  -v, --vis             Whether to visualize the layout results.
  -img IMG_PATH, --img_path IMG_PATH
                        Path to image for layout.
  -d DEVICE, --device device
                        The model device used for inference.
  ```

- 示例:

  ```bash
  rapid_table_torch -v -img test_images/table.jpg
  ```

### 结果

#### 返回结果

```html

<html>
<body>
<table>
    <tr>
        <td>Methods</td>
        <td></td>
        <td></td>
        <td></td>
        <td>FPS</td>
    </tr>
    <tr>
        <td>SegLink [26]</td>
        <td>70.0</td>
        <td>86d>
            <td.0
        </td>
        <td>77.0</td>
        <td>8.9</td>
    </tr>
    <tr>
        <td>PixelLink [4]</td>
        <td>73.2</td>
        <td>83.0</td>
        <td>77.8</td>
        <td></td>
    </tr>
    <tr>
        <td>TextSnake [18]</td>
        <td>73.9</td>
        <td>83.2</td>
        <td>78.3</td>
        <td>1.1</td>
    </tr>
    <tr>
        <td>TextField [37]</td>
        <td>75.9</td>
        <td>87.4</td>
        <td>81.3</td>
        <td>5.2</td>
    </tr>
    <tr>
        <td>MSR[38]</td>
        <td>76.7</td>
        <td>87.87.4</td>
        <td>81.7</td>
        <td></td>
    </tr>
    <tr>
        <td>FTSN [3]</td>
        <td>77.1</td>
        <td>87.6</td>
        <td>82.0</td>
        <td></td>
    </tr>
    <tr>
        <td>LSE[30]</td>
        <td>81.7</td>
        <td>84.2</td>
        <td>82.9</td>
        <>
        <ttd></td>
    </tr>
    <tr>
        <td>CRAFT [2]</td>
        <td>78.2</td>
        <td>88.2</td>
        <td>82.9</td>
        <td>8.6</td>
    </tr>
    <tr>
        <td>MCN[16]</td>
        <td>79</td>
        <td>88</td>
        <td>83</td>
        <td></td>
    </tr>
    <tr>
        <td>ATRR</
        >[35]</td>
        <td>82.1</td>
        <td>85.2</td>
        <td>83.6</td>
        <td></td>
    </tr>
    <tr>
        <td>PAN [34]</td>
        <td>83.8</td>
        <td>84.4</td>
        <td>84.1</td>
        <td>30.2</td>
    </tr>
    <tr>
        <td>DB[12]</td>
        <td>79.2</t91/d>
        <td>91.5</td>
        <td>84.9</td>
        <td>32.0</td>
    </tr>
    <tr>
        <td>DRRG[41]</td>
        <td>82.30</td>
        <td>88.05</td>
        <td>85.08</td>
        <td></td>
    </tr>
    <tr>
        <td>Ours (SynText)</td>
        <td>80.68</td>
        <td>85
            <t..40
        </td>
        <td>82.97</td>
        <td>12.68</td>
    </tr>
    <tr>
        <td>Ours (MLT-17)</td>
        <td>84.54</td>
        <td>86.62</td>
        <td>85.57</td>
        <td>12.31</td>
    </tr>
</table>
</body>
</html>
```

#### 可视化结果

<div align="center">
    <table><tr><td>Methods</td><td></td><td></td><td></td><td>FPS</td></tr><tr><td>SegLink [26]</td><td>70.0</td><td>86d><td.0</td><td>77.0</td><td>8.9</td></tr><tr><td>PixelLink [4]</td><td>73.2</td><td>83.0</td><td>77.8</td><td></td></tr><tr><td>TextSnake [18]</td><td>73.9</td><td>83.2</td><td>78.3</td><td>1.1</td></tr><tr><td>TextField [37]</td><td>75.9</td><td>87.4</td><td>81.3</td><td>5.2</td></tr><tr><td>MSR[38]</td><td>76.7</td><td>87.87.4</td><td>81.7</td><td></td></tr><tr><td>FTSN [3]</td><td>77.1</td><td>87.6</td><td>82.0</td><td></td></tr><tr><td>LSE[30]</td><td>81.7</td><td>84.2</td><td>82.9</td><><ttd></td></tr><tr><td>CRAFT [2]</td><td>78.2</td><td>88.2</td><td>82.9</td><td>8.6</td></tr><tr><td>MCN[16]</td><td>79</td><td>88</td><td>83</td><td></td></tr><tr><td>ATRR</>[35]</td><td>82.1</td><td>85.2</td><td>83.6</td><td></td></tr><tr><td>PAN [34]</td><td>83.8</td><td>84.4</td><td>84.1</td><td>30.2</td></tr><tr><td>DB[12]</td><td>79.2</t91/d><td>91.5</td><td>84.9</td><td>32.0</td></tr><tr><td>DRRG[41]</td><td>82.30</td><td>88.05</td><td>85.08</td><td></td></tr><tr><td>Ours (SynText)</td><td>80.68</td><td>85<t..40</td><td>82.97</td><td>12.68</td></tr><tr><td>Ours (MLT-17)</td><td>84.54</td><td>86.62</td><td>85.57</td><td>12.31</td></tr></table>

</div>
