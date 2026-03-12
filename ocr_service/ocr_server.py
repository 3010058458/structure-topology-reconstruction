import os
import sys

# ⚠️ 必须在导入 paddle 之前设置这些环境变量！
# 禁用 oneDNN 以避免兼容性问题
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLE_USE_MKLDNN'] = '0'
os.environ['FLAGS_mkldnn_enabled'] = '0'

# 禁用 PIR 模式，使用传统执行器
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'

# 使用传统的 IR executor
os.environ['FLAGS_PIR_EXECUTOR'] = '0'

from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import uuid
import threading
import time

app = Flask(__name__)

# 使用递归锁（RLock）而不是普通锁
ocr_lock = threading.RLock()
ocr_instance = None
ocr_last_error_time = 0
ocr_error_count = 0

def rotate_image_high_quality(image_path, angle):
    """
    高质量旋转图片

    Args:
        image_path: 原图路径
        angle: 旋转角度 (90, 180, 270)

    Returns:
        旋转后的图片路径
    """
    import cv2
    import numpy as np

    # 读取原图（保持最高质量）
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    # 根据角度旋转
    # 注意：PaddleOCR 的 angle 表示图片需要逆时针旋转的角度
    if angle == 90:
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return image_path  # 不需要旋转

    # 保存旋转后的图片（无压缩）
    base_name = os.path.splitext(image_path)[0]
    rotated_path = f"{base_name}_rotated_hq.png"

    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    is_success, im_buf_arr = cv2.imencode(".png", rotated, encode_params)
    if is_success:
        im_buf_arr.tofile(rotated_path)
        return rotated_path

    return None

def get_ocr(force_reload=False):
    """获取或创建 OCR 实例"""
    global ocr_instance, ocr_error_count, ocr_last_error_time

    if force_reload or ocr_instance is None:
        from paddleocr import PaddleOCR
        print(f"[{time.strftime('%H:%M:%S')}] 正在初始化 OCR 实例...")
        # 禁用预处理功能以兼容 PaddlePaddle 3.2.0
        # 注意：使用 PaddlePaddle 3.2.0 可以避免 3.3.0 中的 PIR/oneDNN bug
        ocr_instance = PaddleOCR(
            lang='ch',
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False
        )
        ocr_error_count = 0
        print(f"[{time.strftime('%H:%M:%S')}] OCR 实例初始化完成（PaddlePaddle 3.2.0 兼容模式）")

    return ocr_instance

def transform_coordinates_back(box, angle, rotated_width, rotated_height, original_width, original_height):
    """
    将旋转后的坐标转换回原图坐标

    Args:
        box: 坐标列表 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (基于旋转后的图片)
        angle: 旋转角度 (0, 90, 180, 270)
        rotated_width: 旋转后图片宽度
        rotated_height: 旋转后图片高度
        original_width: 原图宽度
        original_height: 原图高度

    Returns:
        转换后的坐标（基于原图）
    """
    import numpy as np

    if not box or len(box) < 4:
        return box

    # 如果没有旋转，直接返回
    if angle == 0:
        return box

    # 转换为numpy数组
    points = np.array(box, dtype=np.float32)

    # 根据旋转角度进行逆变换
    if angle == 90:
        # 原图顺时针旋转90度: 原图(W,H) -> 旋转后(H,W)
        # 逆变换公式: 旋转后(x',y') -> 原图(original_width - y', x')
        new_points = []
        for x, y in points:
            new_x = float(original_width - y)
            new_y = float(x)
            new_points.append([new_x, new_y])
        return new_points

    elif angle == 180:
        # 旋转180度
        # 坐标映射: 旋转后(x',y') -> 原图(W-x', H-y')
        new_points = []
        for x, y in points:
            new_x = float(original_width - x)
            new_y = float(original_height - y)
            new_points.append([new_x, new_y])
        return new_points

    elif angle == 270:
        # 原图顺时针旋转270度: 原图(W,H) -> 旋转后(H,W)
        # 坐标映射: 旋转后(x',y') -> 原图(H-y', x')
        new_points = []
        for x, y in points:
            new_x = float(original_height - y)
            new_y = float(x)
            new_points.append([new_x, new_y])
        return new_points

    return box

def parse_ocr_result(result, original_img_path=None):
    """解析 OCR 结果，如果图片被旋转则保存旋转后的图片"""
    formatted_result = []

    if not result:
        return formatted_result

    # 获取旋转信息
    rotation_angle = 0
    rotated_img = None
    rotated_img_path = None

    if original_img_path:
        import cv2
        import numpy as np
        import os

        # 读取原图
        img = cv2.imdecode(np.fromfile(original_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            img_height, img_width = img.shape[:2]
            print(f"[{time.strftime('%H:%M:%S')}] 原图尺寸: {img_width}x{img_height}")

    # 新版本：返回字典列表
    if isinstance(result, list) and len(result) > 0:
        for page in result:
            print(f"[{time.strftime('%H:%M:%S')}] Page类型: {type(page)}, 类名: {page.__class__.__name__}")

            # 尝试获取旋转角度和旋转后的图片
            try:
                if 'doc_preprocessor_res' in page:
                    doc_res = page['doc_preprocessor_res']
                    print(f"[{time.strftime('%H:%M:%S')}] 通过字典访问到doc_preprocessor_res")

                    if 'angle' in doc_res:
                        rotation_angle = doc_res['angle']
                        print(f"[{time.strftime('%H:%M:%S')}] 检测到图片旋转角度: {rotation_angle}度")

                        # 获取 PaddleOCR 旋转后的图片
                        if rotation_angle != 0 and 'output_img' in doc_res:
                            rotated_img = doc_res['output_img']

                            # 保存旋转后的图片（使用无压缩 PNG）
                            if original_img_path and rotated_img is not None:
                                import cv2
                                base_name = os.path.splitext(original_img_path)[0]
                                rotated_img_path = f"{base_name}_rotated.png"

                                # 使用无压缩保存，保持质量
                                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
                                is_success, im_buf_arr = cv2.imencode(".png", rotated_img, encode_params)
                                if is_success:
                                    im_buf_arr.tofile(rotated_img_path)
                                    print(f"[{time.strftime('%H:%M:%S')}] 已保存旋转后的图片到: {rotated_img_path}")
                                    print(f"[{time.strftime('%H:%M:%S')}] 图片尺寸: {rotated_img.shape[1]}x{rotated_img.shape[0]}")

            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 获取旋转信息异常: {e}")

            # 检查是否有rec_texts（新版本格式）
            if hasattr(page, 'rec_texts') and hasattr(page, 'rec_scores'):
                texts = page.rec_texts
                scores = page.rec_scores
                boxes = page.rec_polys if hasattr(page, 'rec_polys') else []

                print(f"[{time.strftime('%H:%M:%S')}] 识别到 {len(texts)} 个文本")
                if rotation_angle != 0:
                    print(f"[{time.strftime('%H:%M:%S')}] 图片已旋转{rotation_angle}度，坐标基于旋转后的图片")

                for i, text in enumerate(texts):
                    confidence = scores[i] if i < len(scores) else 1.0
                    box = boxes[i].tolist() if i < len(boxes) and hasattr(boxes[i], 'tolist') else []

                    # 不做坐标转换，直接使用OCR返回的坐标
                    formatted_result.append({
                        "text": text,
                        "confidence": float(confidence),
                        "box": box
                    })

            # 兼容字典格式
            elif isinstance(page, dict) and 'rec_texts' in page and 'rec_scores' in page:
                texts = page['rec_texts']
                scores = page['rec_scores']
                boxes = page.get('rec_polys', [])

                for i, text in enumerate(texts):
                    confidence = scores[i] if i < len(scores) else 1.0
                    box = boxes[i].tolist() if i < len(boxes) and hasattr(boxes[i], 'tolist') else []

                    # 不做坐标转换，直接使用OCR返回的坐标
                    formatted_result.append({
                        "text": text,
                        "confidence": float(confidence),
                        "box": box
                    })

    # 旧版本：返回嵌套列表
    elif isinstance(result, list) and len(result) > 0:
        for page in result:
            if isinstance(page, dict):
                continue
            elif isinstance(page, list):
                for line in page:
                    if isinstance(line, str):
                        formatted_result.append({
                            "text": line,
                            "confidence": 1.0,
                            "box": []
                        })
                    elif isinstance(line, (list, tuple)) and len(line) >= 2:
                        box = line[0]
                        text_info = line[1]

                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                        elif isinstance(text_info, str):
                            text = text_info
                            confidence = 1.0
                        else:
                            continue

                        if hasattr(box, 'tolist'):
                            box = box.tolist()

                        formatted_result.append({
                            "text": text,
                            "confidence": float(confidence),
                            "box": box
                        })

    return formatted_result

def do_ocr_with_retry(temp_path, max_retries=2):
    """
    执行 OCR，带重试机制
    """
    global ocr_error_count, ocr_last_error_time

    for attempt in range(max_retries):
        try:
            # 使用递归锁保护
            with ocr_lock:
                # 如果连续错误太多，强制重新加载
                force_reload = (ocr_error_count >= 2)

                ocr = get_ocr(force_reload=force_reload)

                # 执行 OCR
                result = ocr.ocr(temp_path)

                # 成功，重置错误计数
                ocr_error_count = 0
                return {"success": True, "result": result}

        except Exception as e:
            error_msg = str(e)
            ocr_error_count += 1
            ocr_last_error_time = time.time()

            print(f"[{time.strftime('%H:%M:%S')}] OCR 错误 (尝试 {attempt + 1}/{max_retries}): {error_msg}")

            # 如果是最后一次尝试，返回错误
            if attempt == max_retries - 1:
                import traceback
                return {
                    "success": False,
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                    "attempt": attempt + 1
                }

            # 否则，重置实例并重试
            with ocr_lock:
                global ocr_instance
                ocr_instance = None

            # 等待一小段时间再重试
            time.sleep(0.5)

    return {"success": False, "error": "OCR 失败，已达到最大重试次数"}

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "message": "OCR service is running",
        "error_count": ocr_error_count
    })

@app.route('/ocr', methods=['POST'])
def ocr_image():
    """OCR 识别接口"""
    temp_path = None
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            temp_path = f'temp_image_{uuid.uuid4().hex}.jpg'
            file.save(temp_path)

        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({"error": "No image data provided"}), 400

            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))

            temp_path = f'temp_image_{uuid.uuid4().hex}.jpg'
            image.save(temp_path)
        else:
            return jsonify({"error": "Invalid request format"}), 400

        # 执行 OCR（带重试）
        result = do_ocr_with_retry(temp_path)

        if result.get('success'):
            formatted_result = parse_ocr_result(result['result'], temp_path)

            # 检查是否生成了旋转图片
            rotated_path = f"{os.path.splitext(temp_path)[0]}_rotated.png"

            response_data = {
                "success": True,
                "result": formatted_result,
                "count": len(formatted_result)
            }

            if os.path.exists(rotated_path):
                response_data["rotated_image"] = rotated_path
                response_data["message"] = "图片已旋转，坐标基于旋转后的图片"

            return jsonify(response_data)
        else:
            return jsonify(result), 500

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.route('/ocr/text-only', methods=['POST'])
def ocr_text_only():
    """简化版 OCR 接口"""
    temp_path = None
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            temp_path = f'temp_image_{uuid.uuid4().hex}.jpg'
            file.save(temp_path)

        elif request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({"error": "No image data provided"}), 400

            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))

            temp_path = f'temp_image_{uuid.uuid4().hex}.jpg'
            image.save(temp_path)
        else:
            return jsonify({"error": "Invalid request format"}), 400

        # 执行 OCR（带重试）
        result = do_ocr_with_retry(temp_path)

        if result.get('success'):
            formatted_result = parse_ocr_result(result['result'], temp_path)
            texts = [item['text'] for item in formatted_result]
            return jsonify({
                "success": True,
                "text": "\n".join(texts)
            })
        else:
            return jsonify(result), 500

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == '__main__':
    print("PaddleOCR 服务启动中...")
    print("使用自动重试机制，提高稳定性")
    print("可用接口:")
    print("  - GET  /health          健康检查")
    print("  - POST /ocr             完整 OCR 识别（返回文字、坐标、置信度）")
    print("  - POST /ocr/text-only   简化 OCR 识别（只返回文字）")
    print("\n服务地址: http://localhost:5000")

    # 预加载 OCR 模型
    print("\n正在预加载 OCR 模型...")
    with ocr_lock:
        get_ocr()
    print("模型加载完成！\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
