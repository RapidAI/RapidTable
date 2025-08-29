<div align="center">
  <div align="center">
    <h1><b>📊 Rapid Table</b></h1>
  </div>

<a href="https://huggingface.co/spaces/RapidAI/TableStructureRec" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Online Demo-blue"></a>
<a href="https://www.modelscope.cn/studios/RapidAI/TableRec/summary" target="_blank"><img src="https://img.shields.io/badge/魔搭-Demo-blue"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.6-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://pypi.org/project/rapid-table/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapid-table"></a>
<a href="https://pepy.tech/project/rapid-table"><img src="https://static.pepy.tech/personalized-badge/rapid-table?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

</div>

### 🌟 简介

RapidTable库是专门用来文档类图像的表格结构还原，表格结构模型均属于序列预测方法，结合RapidOCR，将给定图像中的表格转化对应的HTML格式。

slanet_plus是paddlex内置的SLANet升级版模型，准确率有大幅提升

unitable是来源unitable的transformer模型，精度最高，暂仅支持pytorch推理，支持gpu推理加速,训练权重来源于 [OhMyTable项目](https://github.com/Sanster/OhMyTable)

### 📅 最近动态

2025-08-29 update: 发布2.1.0，支持batch推理
2025-06-22 update: 发布v2.x，适配rapidocr v3.x \
2025-01-09 update: 发布v1.x，全新接口升级。 \
2024.12.30 update：支持Unitable模型的表格识别，使用pytorch框架 \
2024.11.24 update：支持gpu推理，适配 rapidOCR 单字识别匹配,支持逻辑坐标返回及可视化 \
2024.10.13 update：补充最新paddlex-SLANet-plus 模型(paddle2onnx原因暂不能支持onnx)

### 📸 效果展示

<div align="center">
    <img src="https://github.com/RapidAI/RapidTable/releases/download/assets/preview.gif" alt="Demo" width="80%" height="80%">
</div>

### 🖥️ 支持设备

通过ONNXRuntime推理引擎支持(`rapid_table>=2.0.0`)：

- DirectML
- 昇腾NPU

具体使用方法：

1. 安装（需要卸载其他onnxruntime）:

    ```bash
    # DirectML
    pip install onnxruntime-directml

    # 昇腾NPU
    pip install onnxruntime-cann
    ```

2. 使用：

    ```python
    from rapidocr import RapidOCR

    from rapid_table import ModelType, RapidTable, RapidTableInput

    # DirectML
    ocr_engine = RapidOCR(params={"EngineConfig.onnxruntime.use_dml": True})
    input_args = RapidTableInput(
        model_type=ModelType.SLANETPLUS, engine_cfg={"use_dml": True}
    )

    # 昇腾NPU
    ocr_engine = RapidOCR(params={"EngineConfig.onnxruntime.use_cann": True})

    input_args = RapidTableInput(
        model_type=ModelType.SLANETPLUS,
        engine_cfg={"use_cann": True, "cann_ep_cfg.gpu_id": 1},
    )

    table_engine = RapidTable(input_args)

    img_path = "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg"

    ori_ocr_res = ocr_engine(img_path)
    ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]

    results = table_engine(img_path, ocr_results=ocr_results)
    results.vis(save_dir="outputs", save_name="vis")
    ```

### 🧩 模型列表

|      `model_type`      |                  模型名称                  | 推理框架 |模型大小 |推理耗时(单图 60KB)|
|:--------------|:--------------------------------------| :------: |:------ |:------ |
|       `ppstructure_en`       | `en_ppstructure_mobile_v2_SLANet.onnx` |   onnxruntime   |7.3M |0.15s |
|       `ppstructure_zh`       | `ch_ppstructure_mobile_v2_SLANet.onnx` |   onnxruntime   |7.4M |0.15s |
| `slanet_plus` |          `slanet-plus.onnx`           |  onnxruntime    |6.8M |0.15s |
| `unitable` |          `unitable(encoder.pth,decoder.pth)` |  pytorch    |500M |cpu(6s) gpu-4090(1.5s)|

模型来源\
[PaddleOCR 表格识别](https://github.com/PaddlePaddle/PaddleOCR/blob/133d67f27dc8a241d6b2e30a9f047a0fb75bebbe/ppstructure/table/README_ch.md)\
[PaddleX-SlaNetPlus 表格识别](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/table_structure_recognition.md)\
[Unitable](https://github.com/poloclub/unitable?tab=readme-ov-file)

模型下载地址：[link](https://www.modelscope.cn/models/RapidAI/RapidTable/files)

### 🛠️ 安装

版本依赖关系如下：

|`rapid_table`|OCR|
|:---:|:---|
|v2.x|`rapidocr>=3.0.0`|
|v1.0.x|`rapidocr>=2.0.0,<3.0.0`|
|v0.x|`rapidocr_onnxruntime`|

由于模型较小，预先将slanet-plus表格识别模型(`slanet-plus.onnx`)打包进了whl包内。其余模型在初始化`RapidTable`类时，会根据`model_type`来自动下载模型到安装包所在`models`目录下。

当然也可以通过`RapidTableInput(model_path='')`来指定自己模型路径（`v1.0.x`  参数变量名使用`model_path`,  `v2.x` 参数变量名变更为`model_dir_or_path`）。注意仅限于我们现支持的`model_type`。

> ⚠️注意：`rapid_table>=v1.0.0`之后，不再将`rapidocr`依赖强制打包到`rapid_table`中。使用前，需要自行安装`rapidocr`包。
>
> ⚠️注意：`rapid_table>=v0.1.0,<1.0.0`之后，不再将`rapidocr`依赖强制打包到`rapid_table`中。使用前，需要自行安装`rapidocr_onnxruntime`包。

```bash
pip install rapidocr
pip install rapid_table

# 基于torch来推理unitable模型
pip install rapid_table[torch] # for unitable inference

# onnxruntime-gpu推理
pip uninstall onnxruntime
pip install onnxruntime-gpu # for onnx gpu inference
```

### 🚀 使用方式

#### 🐍 Python脚本运行

> ⚠️注意：在`rapid_table>=1.0.0`之后，模型输入均采用dataclasses封装，简化和兼容参数传递。输入和输出定义如下：

ModelType支持已有的4个模型 ([source](./rapid_table/utils/typings.py))：

```python
class ModelType(Enum):
    PPSTRUCTURE_EN = "ppstructure_en" # onnxruntime
    PPSTRUCTURE_ZH = "ppstructure_zh" # onnxruntime
    SLANETPLUS = "slanet_plus"  # onnxruntime
    UNITABLE = "unitable"   # torch推理引擎
```

#### batch_size推理

```python
from pathlib import Path

from rapid_table import ModelType, RapidTable, RapidTableInput

input_args = RapidTableInput(model_type=ModelType.PPSTRUCTURE_ZH)
table_engine = RapidTable(input_args)

img_list = list(Path("images").iterdir())
results = table_engine(img_path, batch_size=3)  # 这里，batch_size默认为1

# indexes：指定可视化的图像索引。默认为0
results.vis(save_dir="outputs", save_name="vis", indexes=(0, 1, 2))
```

##### CPU使用

```python
from rapid_table import ModelType, RapidTable, RapidTableInput

input_args = RapidTableInput(model_type=ModelType.UNITABLE)
table_engine = RapidTable(input_args)

img_path = "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg"
ori_ocr_res = ocr_engine(img_path)
results = table_engine(img_path)
results.vis(save_dir="outputs", save_name="vis")
```

##### GPU使用

> `engine_cfg`中参数是和[`engine_cfg.yaml`](https://github.com/RapidAI/RapidTable/blob/6da3974a35ac5da8a5cf58194eab00b6886212e8/rapid_table/engine_cfg.yaml)相对应的。

```python
from rapid_table import ModelType, RapidTable, RapidTableInput

# onnxruntime-gpu
input_args = RapidTableInput(
    model_type=ModelType.SLANETPLUS,
    engine_cfg={"use_cuda": True, "cuda_ep_cfg.gpu_id": 1}
)

# torch gpu
# input_args = RapidTableInput(
#     model_type=ModelType.UNITABLE,
#     engine_cfg={"use_cuda": True, "gpu_id": 1},
# )

table_engine = RapidTable(input_args)

img_path = "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg"
results = table_engine(img_path)
results.vis(save_dir="outputs", save_name="vis")
```

#### 📦 终端运行

```bash
rapid_table https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg -v
```

### 📝 结果

#### 📎 返回结果

<details>

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

</details>

#### 🖼️ 可视化结果

<div align="center">
    <table><tr><td>Methods</td><td></td><td></td><td></td><td>FPS</td></tr><tr><td>SegLink [26]</td><td>70.0</td><td>86d><td.0</td><td>77.0</td><td>8.9</td></tr><tr><td>PixelLink [4]</td><td>73.2</td><td>83.0</td><td>77.8</td><td></td></tr><tr><td>TextSnake [18]</td><td>73.9</td><td>83.2</td><td>78.3</td><td>1.1</td></tr><tr><td>TextField [37]</td><td>75.9</td><td>87.4</td><td>81.3</td><td>5.2</td></tr><tr><td>MSR[38]</td><td>76.7</td><td>87.87.4</td><td>81.7</td><td></td></tr><tr><td>FTSN [3]</td><td>77.1</td><td>87.6</td><td>82.0</td><td></td></tr><tr><td>LSE[30]</td><td>81.7</td><td>84.2</td><td>82.9</td><><ttd></td></tr><tr><td>CRAFT [2]</td><td>78.2</td><td>88.2</td><td>82.9</td><td>8.6</td></tr><tr><td>MCN[16]</td><td>79</td><td>88</td><td>83</td><td></td></tr><tr><td>ATRR</>[35]</td><td>82.1</td><td>85.2</td><td>83.6</td><td></td></tr><tr><td>PAN [34]</td><td>83.8</td><td>84.4</td><td>84.1</td><td>30.2</td></tr><tr><td>DB[12]</td><td>79.2</t91/d><td>91.5</td><td>84.9</td><td>32.0</td></tr><tr><td>DRRG[41]</td><td>82.30</td><td>88.05</td><td>85.08</td><td></td></tr><tr><td>Ours (SynText)</td><td>80.68</td><td>85<t..40</td><td>82.97</td><td>12.68</td></tr><tr><td>Ours (MLT-17)</td><td>84.54</td><td>86.62</td><td>85.57</td><td>12.31</td></tr></table>

</div>

### 🔄 与[TableStructureRec](https://github.com/RapidAI/TableStructureRec)关系

TableStructureRec库是一个表格识别算法的集合库，当前有`wired_table_rec`有线表格识别算法和`lineless_table_rec`无线表格识别算法的推理包。

RapidTable是整理自PP-Structure中表格识别部分而来。由于PP-Structure较早，这个库命名就成了`rapid_table`。

总之，RapidTable和TabelStructureRec都是表格识别的仓库。大家可以都试试，哪个好用用哪个。由于每个算法都不太同，暂时不打算做统一处理。

关于表格识别算法的比较，可参见[TableStructureRec测评](https://github.com/RapidAI/TableStructureRec#指标结果)

### 📌 更新日志 ([more](https://github.com/RapidAI/RapidTable/releases))

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
