# Structure Topology Skill

**从建筑结构图纸中自动提取结构信息的 AI Skill**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 概述

本 Skill 面向建筑结构工程场景，能够从结构图纸（立面图、平面图）中自动提取楼层标高、轴网坐标、柱梁墙位置等关键信息，输出标准化 JSON，可直接用于三维结构拓扑重建。

### 核心特性

| 特性 | 说明 |
|------|------|
| **两类图纸** | 立面图（楼层标高）+ 平面图（柱/梁/墙） |
| **双模型交叉验证** | Gemini 3.1 Pro（主）+ Qwen3.5-397B（辅），两阶段迭代协商，提升准确率 |
| **两阶段平面图提取** | 先提取轴网，再注入轴网提取构件，减少坐标偏差 |
| **OCR 增强** | PaddleOCR 识别文字标注，注入 LLM Prompt，提升标高/轴号识别精度 |
| **自动容错** | OCR 服务崩溃自动重启 + 重试；大图自动压缩；JSON 解析失败自动修复 |
| **PDF 支持** | 600 DPI 高清转换，每页独立处理 |

### 评测成绩

在 22 张标注图纸（6 立面 + 2 复杂平面 + 5 中等平面 + 9 简单平面）上，本 Skill 与 9 个裸 LLM baseline 对比：

| 排名 | 系统 | 综合加权分 |
|:----:|------|:---------:|
| 🥇 **1** | **本 Skill** | **0.9305** |
| 2 | Gemini 3.1 Pro（裸调用） | 0.8270 |
| 3 | GPT-5.4-pro（裸调用） | 0.8113 |

> 难度权重：复杂平面图 ×3 / 中等平面图 ×1.5 / 立面图·简单平面图 ×1

---

## 快速开始

### 1. 环境要求

- Python 3.9+
- conda 环境（推荐）或 venv

### 2. 安装依赖

```bash
# 克隆仓库
git clone <repo-url>
cd skill_for_coze

# 创建 conda 环境（推荐）
conda create -n skills python=3.9 -y
conda activate skills

# 安装核心依赖
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
# 复制模板
cp .env.example .env

# 编辑 .env，填入你的 OpenRouter API Key
# 申请地址：https://openrouter.ai/keys
OPENROUTER_API_KEY=your_key_here
```

然后在使用前加载环境变量：

```bash
# Linux / macOS
export $(cat .env | xargs)

# Windows PowerShell
Get-Content .env | ForEach-Object { $k,$v = $_ -split '=',2; [System.Environment]::SetEnvironmentVariable($k, $v) }
```

### 4. 启动 OCR 服务（可选，提升准确率）

```bash
cd ocr_service
python ocr_server.py
# 等待约 20 秒让模型加载完成
# 验证：curl http://localhost:5000/health
```

> 不启动 OCR 时使用 `--no-ocr` 参数，仅靠 Vision LLM 识别，数值精度略低。

### 5. 处理图纸

```bash
cd scripts

# 处理单张图纸
python process_drawings.py --images ../demo/input/立面图_test.png --output ./output

# 处理整个目录
python process_drawings.py --input-dir ../demo/input --output ./output

# 处理 PDF（自动逐页转图）
python process_drawings.py --images drawing.pdf --output ./output

# 禁用 OCR（无 OCR 服务时）
python process_drawings.py --images drawing.png --output ./output --no-ocr

# 禁用交叉验证（更快，精度略低）
python process_drawings.py --images drawing.png --output ./output --no-cross-validation
```

---

## 输出格式

每张图纸生成一个 `{图片名}_extraction.json`。

### 立面图示例

```json
{
  "drawing_type": "elevation",
  "floor_id": "立面图",
  "data": {
    "floor_levels": [
      {"floor": "1F", "elevation": 0.0,    "floor_height": 3600.0},
      {"floor": "2F", "elevation": 3600.0, "floor_height": 3600.0},
      {"floor": "3F", "elevation": 7200.0, "floor_height": 3200.0},
      {"floor": "RF", "elevation": 10400.0,"floor_height": null}
    ],
    "total_height": 10400.0,
    "floor_count": 3
  },
  "ocr_used": true
}
```

### 平面图示例

```json
{
  "drawing_type": "plan",
  "floor_id": "1F",
  "data": {
    "grid_info": {
      "x_axes": [{"label": "1", "coordinate": 0}, {"label": "2", "coordinate": 6000}],
      "y_axes": [{"label": "A", "coordinate": 0}, {"label": "B", "coordinate": 6000}]
    },
    "components_above": {
      "columns": [
        {"x": 0, "y": 0, "label": "KZ1", "grid_location": "A-1", "section": "400x400"},
        {"x": 6000, "y": 0, "label": "KZ1", "grid_location": "A-2", "section": "400x400"}
      ],
      "beams": [
        {"start_grid": "A-1", "end_grid": "A-2", "start": [0, 0], "end": [6000, 0], "label": "KL1", "section": "250x500"}
      ],
      "walls": []
    }
  },
  "ocr_used": true
}
```

**坐标系**：原点 = 编号最小数字轴 × 最小字母轴交点，X 轴为数字轴方向，Y 轴为字母轴方向，单位 mm。

---

## 三维重建

提取完成后，可将多张图纸的 JSON 合并生成 VTU 三维模型，在 ParaView 中可视化：

```bash
cd scripts
pip install pyvista  # 按需安装

# 传入输出目录，自动合并立面图 + 各楼层平面图
python json_to_vtu.py --input ./output --vtu model.vtu
```

用 [ParaView](https://www.paraview.org/download/) 打开 `.vtu`，选 `component_type` 着色（0=柱 / 1=梁 / 2=墙）。

---

## 评测体系

本仓库包含完整的评测框架，可复现论文结果或评测其他模型：

```
evaluation/
├── datasets/
│   ├── images/           # 22 张测试图纸（立面 + 复杂/中等/简单平面图）
│   ├── ground_truth/     # 人工标注的 GT JSON（22 份）
│   ├── out_json/
│   │   ├── skill_result/ # 本 Skill 的输出结果
│   │   └── reports/      # 最新评测报告
│   └── metadata.csv      # 图纸元信息
└── runners/
    ├── scorer.py         # 单模型评分（E1/E2/E3/G1/G2/G3/C1/C2 指标）
    └── compare.py        # 多模型横向对比，输出排名 CSV
```

### 对本 Skill 重新评分

```bash
cd evaluation/runners
python compare.py
# 输出：out_json/reports/comparison_summary.csv + comparison_detail.csv
```

### 评分指标说明

**立面图**（综合权重 = type_id×10% + E1×30% + E2×40% + E3×20%）

| 代号 | 指标 | 说明 |
|------|------|------|
| E1 | 楼层数误差 | 误差=0 得满分 |
| E2 | 标高命中率 + MAE | 容差 ±10mm |
| E3 | 层高正确率 | 容差 ±20mm |

**平面图**（综合权重 = type_id×10% + G1×15% + G2×15% + G3×20% + C1×15% + C2×25%）

| 代号 | 指标 | 说明 |
|------|------|------|
| G1 | 轴网数量 | X/Y 轴数均正确得满分 |
| G2 | 轴线标签命中率 | 精确匹配 |
| G3 | 轴网间距命中率 | 容差 ±50mm |
| C1 | 柱数量误差 | — |
| C2 | 柱位置 F1 | 基于 grid_location 精确匹配 |

---

## 架构说明

```
用户图纸（PNG/JPG/PDF）
    │
    ├─ PDF → 图片（600 DPI，pymupdf）
    │
    ├─ 图像预处理（去噪/增强，仅供 OCR）
    │
    ├─ OCR 识别（PaddleOCR，置信度 ≥ 0.80）
    │
    ├─ 交叉验证：图纸类型识别（Gemini + Qwen）
    │
    ├─ 平面图两阶段提取：
    │   ├─ 阶段1：交叉验证提取轴网（x_axes / y_axes）
    │   └─ 阶段2：注入轴网，交叉验证提取构件（柱/梁/墙）
    │              + 梁坐标后处理（start_grid → 数值坐标）
    │
    ├─ 立面图单阶段：交叉验证提取楼层标高
    │
    └─ JSON 输出
```

### 模块说明

| 文件 | 职责 |
|------|------|
| `scripts/process_drawings.py` | CLI 入口，支持图片/目录/PDF |
| `scripts/enhanced_image_processor.py` | 增强批处理器（交叉验证 + 上下文管理） |
| `scripts/image_processor.py` | 基础处理器，构建 Prompt、调用 LLM、解析 JSON |
| `scripts/cross_validation.py` | 双模型交叉验证（最多 3 轮迭代协商） |
| `scripts/context_manager.py` | 跨图纸会话上下文持久化 |
| `scripts/client_interfaces.py` | OCR/LLM 客户端抽象；大图自动压缩；OCR 500 自动重启 |
| `scripts/pdf_to_image.py` | PDF → PNG（600 DPI） |
| `scripts/image_preprocessor.py` | 图像预处理（去噪/对比度增强） |
| `scripts/json_to_vtu.py` | JSON → VTU 三维模型 |
| `ocr_service/ocr_server.py` | PaddleOCR Flask HTTP 服务（端口 5000） |

---

## 配置参考

`config.json` 关键配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ocr.confidence_threshold` | `0.80` | OCR 置信度过滤阈值 |
| `llm.model` | `google/gemini-3.1-pro-preview` | 主模型 |
| `llm.max_tokens` | `32768` | 最大输出 token 数 |
| `llm.reasoning_enabled` | `true` | 推理模式（仅 Gemini 3.1 Pro Preview） |
| `cross_validation.enabled` | `true` | 是否启用双模型交叉验证 |
| `cross_validation.max_validation_rounds` | `3` | 最大验证轮数 |
| `processing.pdf_dpi` | `600` | PDF 转图像分辨率 |

---

## 常见问题

**Q: OCR 服务返回 500 怎么办？**
A: 通常是 PaddlePaddle 版本问题。必须使用 `paddlepaddle==3.2.0`，3.3.0 有已知 oneDNN bug 导致所有请求返回 500。运行 `pip install paddlepaddle==3.2.0` 修复。

**Q: HTTP 代理导致 OCR 服务无法访问？**
A: 若本机配置了代理（如 Clash），需在启动脚本前临时关闭代理，或将 `localhost` 加入 `NO_PROXY`。代码层面已对所有本地请求添加 `proxies={"http": None}` 绕过。

**Q: 处理超时怎么办？**
A: 单张图纸（含交叉验证）最长约 15 分钟。确保调用时 timeout 设置充足（建议 1200 秒）。

**Q: 识别结果柱数量偏少？**
A: 已知限制：Transformer attention 不擅长逐像素计数密集同类符号。Prompt 中已加入逐轴网交叉点扫描指令，可改善但无法完全消除。如需精确数量，建议配合人工审核。

**Q: 如何禁用双模型交叉验证？**
A: `--no-cross-validation` 或在 `config.json` 中设 `cross_validation.enabled: false`，改为单模型单次处理，速度提升约 50%。

---

## 已知限制

| 能力 | 可靠性 | 说明 |
|------|:------:|------|
| 图纸类型判断（平面/立面） | 高 | 评测准确率 100% |
| 轴线标签、楼层标高读取 | 高 | OCR + LLM 双重验证 |
| 精确统计构件数量 | 中 | LLM 有漏数风险，复杂图纸更明显 |
| 构件 mm 级精确坐标 | 中 | 坐标基于轴网推算，非直接测量 |

---

## 依赖说明

```
requests==2.32.5       # HTTP 请求
pillow==9.5.0          # 图像处理
opencv-python==4.11.0.86  # 图像预处理
numpy==1.26.4          # 数值计算
pymupdf==1.27.1        # PDF → PNG

# OCR 服务（可选，--no-ocr 时不需要）
flask==3.1.3
paddlepaddle==3.2.0    # 严格锁定！3.3.0 有 oneDNN bug
paddleocr==3.4.0

# 三维重建（可选，json_to_vtu.py 使用）
pyvista==0.47.1
```

---

## License

MIT License — 详见 [LICENSE](LICENSE)
