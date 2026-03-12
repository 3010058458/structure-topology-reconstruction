#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PaddleOCR 命令行工具
支持 OCR 识别和置信度可视化
"""

import requests
import sys
import os
import argparse
import cv2
import numpy as np

def ocr_image(image_path, mode='full', server_url='http://localhost:5000'):
    """
    OCR 识别图片

    Args:
        image_path: 图片路径
        mode: 识别模式 'full' 或 'text'
        server_url: 服务器地址

    Returns:
        识别结果字典
    """
    if not os.path.exists(image_path):
        return {'success': False, 'error': f'文件不存在: {image_path}'}

    if not os.path.isfile(image_path):
        return {'success': False, 'error': f'不是有效的文件: {image_path}'}

    try:
        endpoint = '/ocr' if mode == 'full' else '/ocr/text-only'
        url = f"{server_url}{endpoint}"

        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=1200)  # 20 分钟

        if response.status_code == 200:
            return response.json()
        else:
            return {
                'success': False,
                'error': f'服务器返回错误: {response.status_code}',
                'details': response.text
            }

    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': '无法连接到 OCR 服务，请确保服务已启动'}
    except Exception as e:
        return {'success': False, 'error': f'发生错误: {str(e)}'}

def draw_boxes(image_path, result, output_path):
    """
    在图片上绘制置信度框

    Args:
        image_path: 原图路径（如果图片被旋转，这里应该是旋转后的图片）
        result: OCR 结果
        output_path: 输出图片路径
    """
    # 读取图片（处理中文路径，使用 COLOR 模式保持质量）
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return False

    print(f"原图尺寸: {img.shape[1]}x{img.shape[0]}, 通道数: {img.shape[2]}")

    # 统计计数器
    high_conf = 0
    mid_conf = 0
    low_conf = 0

    # 遍历所有识别结果
    for item in result['result']:
        confidence = item['confidence']
        box = item['box']

        if not box or len(box) < 4:
            continue

        # 根据置信度选择颜色 (BGR格式)
        if confidence >= 0.90:
            color = (255, 0, 0)  # 蓝色
            thickness = 2
            high_conf += 1
        elif confidence >= 0.75:
            color = (128, 0, 128)  # 紫色
            thickness = 2
            mid_conf += 1
        else:
            color = (0, 0, 255)  # 红色
            thickness = 3
            low_conf += 1

        # 绘制多边形边框
        points = np.array(box, dtype=np.int32)
        cv2.polylines(img, [points], True, color, thickness)

        # 标注低置信度和中置信度的百分比
        if confidence < 0.90:
            text_label = f"{confidence*100:.1f}%"
            min_y = min([p[1] for p in box])
            min_y_points = [p for p in box if p[1] == min_y]
            text_pos = (int(min_y_points[0][0]), int(min_y) - 5)
            cv2.putText(img, text_label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 1, cv2.LINE_AA)

    # 保存结果（处理中文路径，使用无压缩 PNG）
    # 设置 PNG 压缩级别为 0（无压缩），保证图片质量
    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    is_success, im_buf_arr = cv2.imencode(".png", img, encode_params)
    if is_success:
        im_buf_arr.tofile(output_path)
        print(f"\n已保存标注图片到: {output_path}")
        print(f"输出图片大小: {len(im_buf_arr) / 1024 / 1024:.2f} MB")
        print(f"\n统计信息:")
        print(f"  总计: {result['count']} 个文本框")
        print(f"  高置信度 (≥90%): {high_conf} 个 (蓝色)")
        print(f"  中置信度 (75-90%): {mid_conf} 个 (紫色)")
        print(f"  低置信度 (<75%): {low_conf} 个 (红色)")
        return True
    else:
        print(f"错误: 无法保存图片到 {output_path}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='PaddleOCR 命令行工具 - 支持 OCR 识别和置信度可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅识别文字
  %(prog)s image.jpg

  # 显示完整信息（文字、坐标、置信度）
  %(prog)s image.jpg --mode full

  # 绘制置信度框并保存
  %(prog)s image.jpg --draw-boxes output.png

  # 指定服务器地址
  %(prog)s image.jpg --server http://192.168.1.100:5000
        """
    )

    parser.add_argument(
        'image_path',
        nargs='+',
        help='图片路径（支持空格）'
    )

    parser.add_argument(
        '-m', '--mode',
        choices=['full', 'text'],
        default='text',
        help='识别模式: full=完整信息, text=仅文字 (默认: text)'
    )

    parser.add_argument(
        '-d', '--draw-boxes',
        metavar='OUTPUT',
        help='绘制置信度框并保存到指定路径'
    )

    parser.add_argument(
        '-s', '--server',
        default='http://localhost:5000',
        help='OCR 服务器地址 (默认: http://localhost:5000)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细信息'
    )

    args = parser.parse_args()

    # 合并路径参数（支持空格）
    image_path = ' '.join(args.image_path).strip('"').strip("'")

    if args.verbose:
        print(f"图片路径: {image_path}")
        print(f"识别模式: {args.mode}")
        print(f"服务器: {args.server}\n")

    # 执行 OCR（如果需要画框，必须使用 full 模式）
    mode = 'full' if args.draw_boxes else args.mode
    result = ocr_image(image_path, mode=mode, server_url=args.server)

    if not result.get('success'):
        print(f"错误: {result.get('error')}")
        if args.verbose and result.get('details'):
            print(f"详情: {result['details']}")
        sys.exit(1)

    # 如果需要画框
    if args.draw_boxes:
        # 确定要标注的图片（如果有旋转后的图片，使用旋转后的）
        if 'rotated_image' in result:
            image_to_mark = result['rotated_image']
            if args.verbose:
                print(f"使用旋转后的图片: {image_to_mark}")
                print(f"提示: {result.get('message', '')}\n")
        else:
            image_to_mark = image_path

        # 绘制框
        if not draw_boxes(image_to_mark, result, args.draw_boxes):
            sys.exit(1)

    # 显示识别结果
    if args.mode == 'text' and not args.draw_boxes:
        print(result['text'])
    elif args.mode == 'full':
        print(f"识别到 {result['count']} 行文字:\n")
        for i, item in enumerate(result['result'], 1):
            print(f"[{i}] {item['text']}")
            print(f"    置信度: {item['confidence']:.4f} ({item['confidence']*100:.2f}%)")
            if item.get('box') and len(item['box']) >= 4:
                box = item['box']
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                print(f"    位置: 左上({min(xs):.0f},{min(ys):.0f}) 右下({max(xs):.0f},{max(ys):.0f})")
            print()

if __name__ == '__main__':
    main()
