---
name: structure-topology-reconstruction
description: 从结构工程图纸提取信息的Skill。采用单张图纸独立处理流程：OCR识别→图纸类型识别→信息提取→JSON输出。支持立面图（提取楼层标高）和平面图（提取柱梁墙）。支持PDF及PNG/JPG/JPEG/BMP/TIFF/GIF/WEBP等格式。
---

# 结构工程图纸信息提取 Skill

## 任务目标
- 本 Skill 用于：从结构工程图纸中提取结构信息（楼层标高、柱梁墙位置）
- 处理流程：对每张图纸独立处理 → OCR 识别 → 图纸类型识别 → 信息提取 → JSON 输出
- 触发条件：用户上传结构图纸并要求提取信息或生成 JSON 数据
- 支持格式：**PDF**、PNG、JPG、JPEG、BMP、TIFF、GIF、WEBP

## 执行环境要求

### Bash 工具超时

> ⚠️ **强制要求：执行本 Skill 的所有 Bash 工具调用，timeout 必须设置为 `1200000`（20 分钟）。忘记设置将导致任务在 2 分钟后被强制中断，图纸处理失败。**

```
timeout: 1200000  // 单位毫秒，20 分钟，不可省略
```

原因：本 Skill 调用 Gemini 3.1 Pro + Opus 4.6 进行双模型交叉验证，单张图纸完整处理耗时约 8-15 分钟，默认 2 分钟超时远不够用。

### 预期处理耗时

| 场景 | 耗时 |
|------|------|
| 单张图纸（无交叉验证差异） | 约 4-6 分钟 |
| 单张图纸（触发多轮交叉验证） | 约 8-15 分钟 |
| PDF 转图像（600 DPI） | 约 3-5 秒/页 |

## 前置准备

### 依赖安装
```
requests>=2.28.0
pillow>=10.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pymupdf>=1.23.0        # PDF 处理（必需）
paddlepaddle>=3.0.0    # OCR 后端（可选）
paddleocr>=2.7.0       # OCR 识别（可选）
flask>=3.0.0           # OCR 服务（可选）
```

### API 密钥
- `OPENROUTER_API_KEY` 环境变量，或在 `config.json` 的 `llm.api_key` 中配置
- 默认使用 `HARD_CODED_OPENROUTER_API_KEY` 作为备用（见 `scripts/client_interfaces.py`）

### OCR 服务管理

OCR 是本 Skill 准确率的关键来源，**每张图纸处理前必须确保 OCR 服务处于健康状态**。

#### 单次使用（普通场景）

**第一步：检查服务是否已启动**
```bash
curl http://localhost:5000/health
# 正常返回：{"status": "ok", "message": "OCR service is running"}
```

**第二步：若未启动，在后台启动**
```bash
cd ocr_service && python ocr_server.py
```

> ⚠️ **启动后必须等待约 15 秒**，让 PaddleOCR 完成模型加载，再执行图纸处理任务。过早调用会导致识别请求失败。

#### 批量处理多张图纸时的强制重启规范

> ⚠️ **强制要求：批量处理时，每处理完一张图纸，必须关闭 OCR 服务并重新启动，再处理下一张。**

原因：PaddleOCR 在连续处理多张大图后容易累积内存占用或进入不稳定状态（表现为 502/500 错误），导致后续图纸的 OCR 识别被跳过，降级为纯 Vision LLM，损失准确率。通过逐图重启可确保每张图纸都能获得干净的 OCR 服务。

**每张图纸的标准处理序列（四步）：**

```bash
# 步骤 1：终止现有 OCR 服务
for /f "tokens=5" %a in ('netstat -ano ^| findstr :5000') do taskkill /PID %a /F

# 步骤 2：重启 OCR 服务（后台）
start /B cmd /c "cd ocr_service && python ocr_server.py"

# 步骤 3：等待 15 秒（模型加载）
timeout /t 15 /nobreak

# 步骤 4：验证健康后处理图纸
curl http://localhost:5000/health
cd scripts && python process_drawings.py --images <图纸路径> --output <输出目录>
```

**验收标准：** 处理完成后日志中应出现 `使用 OCR: 1/1`，若显示 `0/1` 则说明 OCR 未生效，需重新处理该图纸。

**500 错误自动恢复（单张图纸内部）：**
- 识别过程中若收到 HTTP 500 响应，`PaddleOCRClient` 会自动：
  1. 终止当前 OCR 进程（kill port 5000）
  2. 重新启动 OCR 服务
  3. 等待 15 秒模型加载
  4. 重试识别一次
- 若重启后仍失败，会抛出异常并记录日志
- 逻辑见 `scripts/client_interfaces.py` → `_restart_service()` 方法

## 操作步骤

### 步骤 0：PDF 转图像（如输入为 PDF）

如果用户提供的是 PDF 文件，需先转换为图像再处理：

```python
from pdf_to_image import PDFToImageConverter

converter = PDFToImageConverter(dpi=600, output_format="png")
image_paths = converter.convert_pdf_to_images(
    pdf_path="图纸.pdf",
    output_dir="./temp_images"
)
# 返回每页对应的图片路径列表，再传入后续步骤
```

> 默认使用 600 DPI（质量优先），如需加快转换速度可降至 300

### 步骤 1：准备输入

用户可能会一次性提交多张图片，包括：
- 立面图：显示楼层标高和竖向构件
- 平面图：显示各楼层的柱梁墙平面布置

**支持的输入方式：**
1. 直接提供图片/PDF 文件路径列表
2. 提供包含图片的目录路径

### 步骤 2：对每张图片独立处理

对于每张图片，按以下流程处理：

#### 2.1 OCR 识别并筛选
- 调用 OCR 服务识别图片中的文字
- 筛选置信度 ≥ 0.85 的识别结果
- 构建文字摘要供后续使用

**实现方式：**
```python
from client_interfaces import create_ocr_client
from image_processor import load_config

config = load_config("config.json")
ocr_client = create_ocr_client(config)

# OCR 识别
ocr_results = ocr_client.recognize(image_path)
# 返回格式：[{"text": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]
```

#### 2.2 第一次 LLM 调用：识别图纸类型
- 使用 Vision LLM 分析图片
- 判断是立面图还是平面图
- 返回图纸类型和置信度

**Prompt 模板：**
```
请分析这张建筑结构图纸，判断它是立面图还是平面图。

判断依据：
- 立面图：显示建筑的侧面视图，包含楼层标高符号（▽）、楼层线、竖向构件
- 平面图：显示建筑的俯视图，包含轴网（数字轴、字母轴）、柱、梁、墙的平面布置

OCR 识别的文本信息：
[OCR 文字摘要]

请返回 JSON 格式：
{
    "drawing_type": "elevation" 或 "plan",
    "confidence": 0.0-1.0 之间的置信度,
    "reasoning": "你的判断理由"
}
```

#### 2.3 第二次 LLM 调用：提取结构信息
- 根据图纸类型选择不同的 Prompt
- 将 OCR 识别结果作为补充信息
- 提取结构化数据

**立面图 Prompt 模板：**
```
你是一名结构工程专家，请从这张建筑立面图中提取楼层标高信息。

结构工程背景知识：
立面图展示建筑的侧面剖面，其中：
- 每条水平楼层线代表一块楼板（floor slab）的位置
- 标高符号（▽）标注的是该楼板面的绝对高度
- 相邻两块楼板之间的空间就是一个"层"，柱、墙等竖向构件就在这个空间内
- 例如：1F 楼板标高 0mm，2F 楼板标高 3600mm，则 1F 的层高 = 3600mm

提取内容：
1. 识别所有标高符号（▽）及其旁边的数值（单位：mm）
2. 识别对应的楼层名称（如 1F、2F、3F、RF 等）
3. 计算相邻楼层之间的层高（floor_height）
4. 按从下到上的顺序排列

OCR 识别的文本信息（可作为参考）：
[OCR 文字摘要]

请返回 JSON 格式：
{
    "floor_id": "立面图",
    "floor_levels": [
        {"floor": "1F", "elevation": 0.0, "floor_height": 3600.0, "description": "一层楼板面标高"},
        {"floor": "2F", "elevation": 3600.0, "floor_height": 3600.0, "description": "二层楼板面标高"},
        {"floor": "RF", "elevation": 7200.0, "floor_height": null, "description": "屋面标高"}
    ],
    "total_height": 7200.0,
    "floor_count": 2,
    "notes": "其他备注信息"
}

注意事项：
1. 标高单位统一为毫米（mm）
2. floor_height = 上一层 elevation - 本层 elevation
3. 最顶层的 floor_height 为 null
4. 按从下到上的顺序排列
```

**平面图 Prompt 模板：**
```
你是一名结构工程专家，请从这张结构平面图中提取构件信息。

结构工程背景知识（非常重要）：
结构平面图（如"1F 结构平面图"）展示的是从该楼层楼板向上延伸的所有结构构件：
- 柱（Column）：竖向构件，从当前楼板面向上延伸到上一层楼板面
  例如：1F 平面图中的柱从 1F 楼板延伸到 2F 楼板
- 梁（Beam）：水平构件，位于上一层楼板的底部
  例如：1F 平面图中的梁位于 2F 楼板底面
- 剪力墙（Shear Wall）：竖向面状构件，从当前楼板面向上延伸到上一层楼板面
- 楼板（Slab）：如有标注，提取范围和厚度

总结：N层平面图中的构件连接关系：
- 柱和墙：底部在 N 层楼板面，顶部在 N+1 层楼板面
- 梁：位于 N+1 层楼板底面

OCR 识别的文本信息（可作为参考）：
[OCR 文字摘要]

坐标系定义：
- 原点：1 轴与 A 轴的交点
- X 轴方向：数字轴增大方向（1→2→3...）
- Y 轴方向：字母轴增大方向（A→B→C...）
- 单位：毫米（mm）

请返回 JSON 格式：
{
    "floor_id": "1F",
    "components_above": {
        "columns": [
            {"x": 0, "y": 0, "label": "KZ1", "grid_location": "A-1", "section": "400x400"}
        ],
        "beams": [
            {"start_grid": "A-1", "end_grid": "A-2", "start": [0, 0], "end": [6000, 0], "label": "KL1", "section": "250x500"}
        ],
        "walls": [
            {"start": [0, 0], "end": [0, 6000], "thickness": 200, "label": "Q1"}
        ],
        "slabs": []
    },
    "grid_info": {
        "x_axes": [{"label": "1", "coordinate": 0}, {"label": "2", "coordinate": 6000}],
        "y_axes": [{"label": "A", "coordinate": 0}, {"label": "B", "coordinate": 6000}]
    },
    "connection_note": "柱和墙从 1F 楼板延伸至 2F 楼板，梁位于 2F 楼板底面",
    "notes": "其他备注信息"
}

注意事项：
1. 坐标必须是基于轴网间距的世界坐标（mm），不是像素坐标
2. 柱的坐标是柱截面中心点
3. 梁和墙的坐标是其轴线位置
4. 如果截面尺寸未标注，设为 null
```

#### 2.4 保存 JSON 文件
- 将提取结果保存为 JSON 文件
- 文件名格式：`{图片名}_extraction.json`

### 步骤 3：处理下一张图片

重复步骤 2，直到所有图片处理完成。

### 步骤 4：返回结果

向用户返回：
- 处理统计信息（立面图数量、平面图数量、总计）
- 输出目录路径
- 每张图片的处理结果摘要

## 使用示例

### 示例 1：使用命令行工具

```bash
# 设置 API 密钥
export OPENROUTER_API_KEY="your-api-key"

# 处理多张图片
cd scripts
python process_drawings.py \
    --images elevation.png 1F.jpg 2F.jpg \
    --output ./output

# 或者处理整个目录
python process_drawings.py \
    --input-dir ../test_images \
    --output ./output
```

### 示例 2：在 Python 脚本中使用

```python
from image_processor import BatchImageProcessor, load_config
from client_interfaces import create_ocr_client, create_llm_client

# 加载配置
config = load_config("config.json")

# 创建客户端
ocr_client = create_ocr_client(config)
llm_client = create_llm_client(config)

# 创建处理器
processor = BatchImageProcessor(
    ocr_client=ocr_client,
    llm_client=llm_client,
    ocr_confidence_threshold=0.85,
    output_dir="./output"
)

# 处理图片
image_paths = ["elevation.png", "1F.jpg", "2F.jpg"]
results = processor.process_images(image_paths)

# 查看结果
for result in results:
    print(f"图纸类型: {result.drawing_type}")
    print(f"楼层: {result.floor_id}")
```

### 示例 3：在 Skill 中使用

Skill 可以实现自己的客户端接口：

```python
from client_interfaces import OCRClientInterface, LLMClientInterface
from image_processor import ImageProcessor

# 实现 Skill 的客户端
class SkillOCRClient(OCRClientInterface):
    def recognize(self, image_path: str):
        # 使用 Skill 的 OCR 能力
        return skill.ocr_recognize(image_path)

class SkillLLMClient(LLMClientInterface):
    def chat(self, prompt: str, image_path=None):
        # 使用 Skill 的 LLM 能力
        return skill.llm_chat(prompt, image_path)

# 创建处理器
processor = ImageProcessor(
    ocr_client=SkillOCRClient(),
    llm_client=SkillLLMClient(),
    ocr_confidence_threshold=0.85,
    output_dir="./output"
)

# 处理用户提交的图片
results = []
for image_path in user_images:
    result = processor.process_image(image_path)
    results.append(result)
```

## 输出格式

### 立面图输出示例

```json
{
  "drawing_type": "elevation",
  "floor_id": "立面图",
  "data": {
    "floor_id": "立面图",
    "floor_levels": [
      {"floor": "1F", "elevation": 0.0, "floor_height": 3600.0, "description": "一层楼板面标高"},
      {"floor": "2F", "elevation": 3600.0, "floor_height": 3600.0, "description": "二层楼板面标高"},
      {"floor": "3F", "elevation": 7200.0, "floor_height": 3200.0, "description": "三层楼板面标高"},
      {"floor": "RF", "elevation": 10400.0, "floor_height": null, "description": "屋面标高"}
    ],
    "total_height": 10400.0,
    "floor_count": 3
  },
  "ocr_used": true,
  "metadata": {
    "image_path": "elevation.png",
    "image_name": "elevation.png",
    "ocr_text_count": 25,
    "type_confidence": 0.95,
    "type_reasoning": "图中包含标高符号和楼层线"
  }
}
```

### 平面图输出示例

```json
{
  "drawing_type": "plan",
  "floor_id": "1F",
  "data": {
    "floor_id": "1F",
    "components_above": {
      "columns": [
        {"x": 0, "y": 0, "label": "KZ1", "grid_location": "A-1", "section": "400x400"},
        {"x": 6000, "y": 0, "label": "KZ1", "grid_location": "A-2", "section": "400x400"}
      ],
      "beams": [
        {"start": [0, 0], "end": [6000, 0], "label": "KL1", "section": "250x500"}
      ],
      "walls": [
        {"start": [0, 0], "end": [0, 6000], "thickness": 200, "label": "Q1"}
      ],
      "slabs": []
    },
    "grid_info": {
      "x_axes": [{"label": "1", "coordinate": 0}, {"label": "2", "coordinate": 6000}],
      "y_axes": [{"label": "A", "coordinate": 0}, {"label": "B", "coordinate": 6000}]
    },
    "connection_note": "柱和墙从 1F 楼板延伸至 2F 楼板，梁位于 2F 楼板底面"
  },
  "ocr_used": true,
  "structural_note": "本图纸（1F 平面图）中的构件连接关系：柱和墙从 1F 楼板面向上延伸至上一层楼板面；梁位于上一层楼板底面。具体的 z 坐标需结合立面图的标高信息确定。",
  "metadata": {
    "image_path": "1F.png",
    "image_name": "1F.png",
    "ocr_text_count": 42,
    "type_confidence": 0.92
  }
}
```

## 拒绝策略

当以下情况触发时，应告知用户图纸质量问题，并跳过该图纸：

1. **OCR 识别失败**：无法连接 OCR 服务或识别结果为空
2. **图纸类型识别失败**：LLM 无法判断图纸类型或置信度过低（< 0.5）
3. **信息提取失败**：LLM 返回的 JSON 格式错误或缺少必要字段
4. **图片格式不支持**：图片格式不在支持列表中

触发拒绝时，应：
- 在控制台输出警告信息
- 继续处理下一张图片
- 在最终统计中标记失败的图片

## 资源索引

| 文件 | 说明 |
|------|------|
| `scripts/process_drawings.py` | 主入口脚本，命令行调用入口 |
| `scripts/json_to_vtu.py` | JSON → VTU 三维重建脚本（需要 pyvista） |
| `scripts/enhanced_image_processor.py` | 增强处理器（交叉验证 + 上下文管理） |
| `scripts/image_processor.py` | 基础图纸处理器 |
| `scripts/client_interfaces.py` | OCR/LLM 客户端接口（含 500 自动重启逻辑） |
| `scripts/pdf_to_image.py` | PDF/图像转换器（支持 PDF 多页转图） |
| `scripts/cross_validation.py` | 双模型交叉验证逻辑 |
| `scripts/context_manager.py` | 多轮对话上下文管理 |
| `scripts/image_preprocessor.py` | 图像预处理（去噪、对比度增强） |
| `scripts/config_validator.py` | 配置文件验证 |
| `scripts/logger.py` | 日志模块 |
| `ocr_service/ocr_server.py` | PaddleOCR HTTP 服务 |
| `ocr_service/ocr_cli.py` | OCR 命令行测试工具 |
| `config.json` | 主配置文件 |
| `CLAUDE.md` | Claude Code 执行规范（超时、OCR 管理、大图压缩） |

## 注意事项
- **Bash 超时**：调用 Bash 工具执行本 Skill 时，**必须将 `timeout` 设置为 `1200000`（20 分钟）**，详见"执行环境要求"章节
- **架构设计**：本 Skill 采用单张图纸独立处理模式，每张图纸调用 2 次 LLM（图纸类型识别 + 信息提取），启用交叉验证时两个模型各调用一次
- **OCR 集成**：OCR 结果作为 Prompt 的补充信息，有助于提升标高数值和轴网标注的识别精度；禁用 OCR 时仅用 Vision LLM（仍可识别，但数值精度略低）
- **PDF 支持**：PDF 需先通过 `PDFToImageConverter` 转换为图像，每页生成一张图片独立处理
- **图纸类型自动识别**：无需用户手动指定图纸类型，置信度低于 0.5 时跳过该图纸
- **坐标系定义**：平面图使用世界坐标系（毫米），原点为 1 轴与 A 轴的交点
- **容错处理**：单张图片处理失败不影响其他图片；OCR 500 错误自动重启服务并重试
- **建筑图 vs 结构图**：模型会区分建筑平面图和结构平面图，建筑图中若无结构构件标注，柱梁列表为空是正确行为

## 配置说明

配置文件 `config.json`：

```json
{
  "project": {
    "version": "5.0.0"
  },
  "ocr": {
    "server_url": "http://localhost:5000",
    "confidence_threshold": 0.80,
    "enabled": true,
    "engine": "PaddleOCR",
    "timeout": 1200
  },
  "llm": {
    "provider": "openrouter",
    "model": "google/gemini-3.1-pro-preview",
    "max_tokens": 4096,
    "temperature": 0.1,
    "reasoning_enabled": true
  },
  "cross_validation": {
    "enabled": true,
    "models": ["google/gemini-3.1-pro-preview", "anthropic/claude-opus-4.6"],
    "max_validation_rounds": 3
  },
  "context_management": {
    "enabled": true,
    "context_dir": "./context",
    "auto_save": true
  },
  "processing": {
    "pdf_dpi": 600
  },
  "image_preprocessing": {
    "enabled": true,
    "for_ocr_only": true
  }
}
```

**关键配置项：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ocr.confidence_threshold` | `0.80` | OCR 置信度阈值，低于此值的识别结果被过滤 |
| `ocr.timeout` | `1200` | OCR 服务请求超时（秒） |
| `llm.provider` | `openrouter` | LLM 提供商（目前仅支持 `openrouter`） |
| `llm.model` | `google/gemini-3.1-pro-preview` | 主模型，支持推理模式 |
| `llm.temperature` | `0.1` | 建议保持低温度以获得确定性输出 |
| `llm.reasoning_enabled` | `true` | 启用推理模式（仅 Gemini 3.1 Pro Preview） |
| `cross_validation.enabled` | `true` | 双模型交叉验证（Gemini + Opus 4.6） |
| `cross_validation.max_validation_rounds` | `3` | 最大验证轮数，超出后使用合并策略 |
| `processing.pdf_dpi` | `600` | PDF 转图像分辨率，默认 600（质量优先） |

## 常见问题

### Q: 如何禁用 OCR？
A: 使用 `--no-ocr` 参数，此时仅使用 Vision LLM 识别。适用于 OCR 服务不可用的场景，识别结果仍可接受。

### Q: 如何处理 PDF 文件？
A: PDF 需先转换为图像。使用 `scripts/pdf_to_image.py` 中的 `PDFToImageConverter`，默认 600 DPI（质量优先）。转换后的图像路径传入处理器即可。

### Q: OCR 服务返回 500 怎么办？
A: 无需手动处理。`PaddleOCRClient` 检测到 500 后会自动终止旧进程、重启服务并重试一次。若重启后仍失败，会抛出异常并记录日志。

### Q: 处理超时怎么办？
A: 确认 Bash 工具的 `timeout` 参数已设置为 `1200000`（20 分钟）。单张图纸含交叉验证最长约 15 分钟。

### Q: 识别出的构件列表为空是否正常？
A: 视情况而定。若图纸为**建筑平面图**（非结构施工图），模型会正确识别为建筑图并将构件列表留空，同时在 `notes` 字段说明需配合结构施工图使用，这是正确行为。

### Q: 如何更换 LLM 模型？
A: 修改 `config.json` 中的 `llm.model`，或启动时使用 `--llm-model` 参数覆盖。

### Q: 交叉验证差异太多导致结果不稳定？
A: 可适当降低 `cross_validation.max_validation_rounds`（如改为 1），或在 `config.json` 中将 `cross_validation.enabled` 设为 `false` 禁用，改用单模型处理。

## 三维重建（JSON → VTU）

提取完成后，可将多张图纸的 JSON 输出合并重建为三维结构模型，在 ParaView 中可视化。

### 前置依赖

```bash
pip install pyvista
```

### 使用方法

```bash
cd scripts

# 方式 1：传入整个输出目录（自动找所有 *_extraction.json）
python json_to_vtu.py --input ../output --vtu model.vtu

# 方式 2：指定具体文件（建议同时传入立面图 + 所有楼层平面图）
python json_to_vtu.py --input 1F.json 2F.json elevation.json --vtu model.vtu
```

### 输入要求

| JSON 类型 | 必需 | 说明 |
|-----------|------|------|
| 立面图（elevation）| 推荐 | 提供各楼层 Z 坐标；缺失时按 3600mm 层高估算 |
| 平面图（plan） | 必需 | 提供柱/梁/墙的 XY 坐标；至少一张 |

### 输出 VTU 结构

| cell_data 字段 | 类型 | 值含义 |
|----------------|------|--------|
| `component_type` | int32 | 0=柱  1=梁  2=剪力墙 |

单元类型：
- 柱：VTK LINE（竖向，从楼板面到上层楼板面）
- 梁：VTK LINE（水平，位于上层楼板高度）
- 剪力墙：VTK QUAD（4 角点面单元）

### ParaView 查看步骤

1. 下载 ParaView：https://www.paraview.org/download/
2. `File → Open` → 选择 VTU 文件 → 点击 `Apply`
3. `Coloring` 下拉框选 `component_type`（0=柱/1=梁/2=墙）
4. 梁柱管状显示：`Filters → Tube` → 调整 Radius
5. 视角：左键旋转、滚轮缩放、快捷键 `R` 重置

> **注意**：坐标单位为毫米（mm）。因 Vision LLM 坐标精度限制，重建模型仅供参考，
> 不可用于结构计算。

## 已知限制

本 Skill 基于 Vision LLM（像素理解），存在以下固有限制，使用前需了解：

| 任务 | 可靠性 | 说明 |
|------|--------|------|
| 图纸类型识别（平面/立面）| 高 | 语义判断，准确率稳定 |
| 读取文字标注、轴线编号 | 高 | OCR + LLM 双重验证 |
| 识别结构体系 | 中 | 宏观判断，不涉及计数 |
| **精确统计构件数量**（如有多少根柱）| **低** | LLM 无法逐像素枚举，结果不可信 |
| **构件精确坐标**（mm 级别）| **低** | LLM 输出坐标是语义估计，不是测量值 |

**根本原因**：Transformer 的 attention 机制擅长语义理解，不擅长系统性遍历计数。图纸中密集排列的同类符号（如 50+ 根柱）会导致漏数或重数。

**适合的使用场景**：
- 快速了解图纸类型和整体结构体系
- 提取标高、轴网标注等文字信息
- 作为辅助工具，工程师在其基础上人工审核

**不适合的使用场景**：
- 需要精确构件数量和坐标的生产级 BIM 数据
- 安全验算所需的结构参数提取

**更准确的替代方案**：
- 若有 DWG/DXF 源文件：解析 `INSERT` 块引用，可精确获取柱数量和坐标
- 若只有 PDF：训练专用目标检测模型（YOLO 等）检测柱符号
