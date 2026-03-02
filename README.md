# Chart2SVG：Tokenizer 更新与训练流程

## 目录与路径说明

- `Chart2SVG/data/`（下文简称 `data/`）：用于存放语义转换、特殊词表（special tokens）的修改与映射。注意修改import中的正确路径。
- 训练脚本依赖 `ms-swift`和 。

## 一、更新 Qwen 模型 Tokenizer

### 操作目的

- 为 Qwen 模型 tokenizer 新增 101 个 special token。

### 执行命令

代码路径需要改成自己的路径。

```bash
python Chart2SVG/models/prepare_svg_qwen3.py
```

## 二、SFT 训练（LoRA 微调）

### 前置说明

- 训练脚本需配置 5 个数据集路径，其中 `echarts - 100+数据量` 和 `fusion - 600+数据量` 数据集可重复填写（数据量较少，用于补充数据量）。

### 操作步骤

1. 前台执行训练（注意替换为 `ms-swift` 对应路径中的脚本；若没有可自行添加）

```bash
bash examples/train/lora_sft.sh
```

2. 后台挂起训练（避免终端关闭中断训练）

```bash
# 日志输出到 training.log，包含标准输出和错误输出
nohup bash examples/train/lora_sft.sh > training.log 2>&1 &
```

3. 实时查看训练日志（监控进度 / 报错）

```bash
tail -f training.log
```

## 三、GRPO 训练

### 关键路径

- 奖励函数路径（替换对应脚本）：`ms-swift-release-3.12/examples/train/grpo/plugin/plugin.py`
- 系统提示词：`prompt.txt`

### 运行

```bash
ms-swift-release-3.12/examples/train/grpo/grpo_qwen.sh
```
