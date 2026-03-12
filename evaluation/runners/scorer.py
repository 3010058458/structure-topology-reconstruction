"""
Evaluation Scorer

读取 baseline_runner.py 产生的原始输出和 ground_truth JSON，
按照 discussion.md §5 定义的指标计算每张图纸的得分，输出 JSON 报告和 Markdown 报告。

评分指标汇总：
  立面图：
    type_id_correct  : 图纸类型是否识别正确（0/1）
    E1_count_error   : |pred_floor_count - gt_floor_count|（0 为满分）
    E1_score         : max(0, 1 - E1_count_error / max(gt_floor_count, 1))
    E2_elevation_mae : 标高 MAE（mm，越低越好）
    E2_hit_rate      : 标高命中率（±10mm 容差内的楼层占比）
    E3_floor_height_accuracy : 层高正确率（±20mm 容差，末层除外）

  平面图：
    type_id_correct  : 图纸类型是否识别正确（0/1）
    G1_x_error       : |pred_x_count - gt_x_count|
    G1_y_error       : |pred_y_count - gt_y_count|
    G1_score         : 1 仅当 x_error=0 且 y_error=0
    G2_label_hit_rate: 轴线标签命中率
    G3_spacing_mae   : 轴网间距 MAE（mm）
    G3_spacing_hit_rate : 间距正确率（±50mm 容差）
    C1_count_error   : |pred_col_count - gt_col_count|
    C1_score         : max(0, 1 - C1_count_error / max(gt_col_count, 1))
    C2_precision     : 柱位置 Precision（基于 grid_location 精确匹配）
    C2_recall        : 柱位置 Recall
    C2_f1            : 柱位置 F1

  综合得分（weighted_score）：
    立面图：type_id×10% + E1×30% + E2_hit_rate×40% + E3×20%
    平面图：type_id×10% + G1×15% + G2×15% + G3_hit×20% + C1×15% + C2_f1×25%

用法：
    cd evaluation/runners

    # 对指定 run_id 打分
    python scorer.py --run-id 20260310_143022_google_gemini-3.1-pro-preview

    # 自动使用最新一次 run
    python scorer.py --latest

    # 指定路径
    python scorer.py --run-dir ./outputs/20260310_143022_google_gemini-3.1-pro-preview \\
                     --gt-dir ../datasets/ground_truth \\
                     --output ./reports

输出：
    reports/{run_id}_scores.json   : 完整评分数据（每张图 + 汇总）
    reports/{run_id}_report.md     : 可读 Markdown 报告
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional


# ============================================================================
# 容差常量（对应 discussion.md §5.2）
# ============================================================================

ELEVATION_TOLERANCE_MM  = 10    # E-2: 标高命中容差
FLOOR_HEIGHT_TOLERANCE  = 20    # E-3: 层高命中容差
SPACING_TOLERANCE_MM    = 50    # G-3: 轴网间距命中容差

# 综合得分权重
WEIGHTS_ELEVATION = {
    "type_id": 0.10,
    "E1":      0.30,
    "E2_hit":  0.40,
    "E3":      0.20,
}
WEIGHTS_PLAN = {
    "type_id": 0.10,
    "G1":      0.15,
    "G2":      0.15,
    "G3_hit":  0.20,
    "C1":      0.15,
    "C2_f1":   0.25,
}


# ============================================================================
# 数据加载
# ============================================================================

def load_gt(gt_dir: Path, image_id: str) -> Optional[dict]:
    """
    加载 GT 文件。GT 文件为 Skill 输出格式（或人工标注后的版本）：
      gt["drawing_type"]  → 图纸类型
      gt["data"]          → 实际数据（floor_levels / grid_info / components_above）

    返回 None 表示文件不存在或 data 为空。
    """
    path = gt_dir / f"{image_id}_extraction.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        gt = json.load(f)
    if not gt.get("data"):
        return None
    return gt


def load_prediction(run_dir: Path, image_id: str) -> Optional[dict]:
    """
    加载 runner 输出文件（{image_id}_extraction.json）。
    返回 None 表示文件不存在。
    """
    path = run_dir / f"{image_id}_extraction.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# 立面图评分
# ============================================================================

def score_elevation(gt_data: dict, pred_data: dict) -> dict:
    """
    计算立面图子任务分数（E-1, E-2, E-3）。

    Args:
        gt_data:   gt["data"] 块
        pred_data: runner 输出 extraction.parsed_output 块（无 data 包装）

    Returns:
        包含所有指标的字典
    """
    metrics = {}

    # ---- E-1: 楼层数 ----
    gt_count   = gt_data.get("floor_count", 0) or 0
    pred_count = pred_data.get("floor_count", 0) or 0
    e1_error   = abs(int(pred_count) - int(gt_count))
    e1_score   = max(0.0, 1.0 - e1_error / max(gt_count, 1))
    metrics["E1_count_error"] = e1_error
    metrics["E1_score"]       = round(e1_score, 4)

    # ---- E-2: 标高准确性 ----
    gt_levels   = gt_data.get("floor_levels", []) or []
    pred_levels = pred_data.get("floor_levels", []) or []

    # 建立 GT floor → elevation 映射
    gt_elev_map = {str(fl["floor"]): float(fl["elevation"])
                   for fl in gt_levels if "floor" in fl and fl.get("elevation") is not None}
    pred_elev_map = {str(fl["floor"]): float(fl["elevation"])
                     for fl in pred_levels if "floor" in fl and fl.get("elevation") is not None}

    errors = []
    hits = 0
    for floor_name, gt_elev in gt_elev_map.items():
        if floor_name in pred_elev_map:
            err = abs(pred_elev_map[floor_name] - gt_elev)
            errors.append(err)
            if err <= ELEVATION_TOLERANCE_MM:
                hits += 1

    metrics["E2_matched_floors"]  = len(errors)
    metrics["E2_gt_floors"]       = len(gt_elev_map)
    metrics["E2_elevation_mae"]   = round(sum(errors) / len(errors), 2) if errors else None
    metrics["E2_hit_rate"]        = round(hits / max(len(gt_elev_map), 1), 4)

    # ---- E-3: 层高正确率（由 E-2 推导，末层除外）----
    # 对 GT 中按 elevation 排序的楼层对，计算相邻层高
    gt_sorted = sorted(gt_levels, key=lambda x: float(x.get("elevation", 0)))
    pred_sorted = sorted(pred_levels, key=lambda x: float(x.get("elevation", 0)))

    gt_spacings = {}   # floor_name → gt_floor_height
    for i in range(len(gt_sorted) - 1):
        cur  = gt_sorted[i]
        nxt  = gt_sorted[i + 1]
        name = str(cur.get("floor", i))
        sp   = float(nxt.get("elevation", 0)) - float(cur.get("elevation", 0))
        gt_spacings[name] = sp

    pred_elev_by_floor = {str(fl.get("floor", "")): float(fl.get("elevation", 0))
                          for fl in pred_levels}

    e3_hits = 0
    e3_total = len(gt_spacings)
    for i in range(len(gt_sorted) - 1):
        cur_name = str(gt_sorted[i].get("floor", ""))
        nxt_name = str(gt_sorted[i + 1].get("floor", ""))
        if cur_name in pred_elev_by_floor and nxt_name in pred_elev_by_floor:
            pred_sp = pred_elev_by_floor[nxt_name] - pred_elev_by_floor[cur_name]
            gt_sp   = gt_spacings.get(cur_name, 0)
            if abs(pred_sp - gt_sp) <= FLOOR_HEIGHT_TOLERANCE:
                e3_hits += 1

    metrics["E3_floor_height_accuracy"] = round(e3_hits / max(e3_total, 1), 4) if e3_total > 0 else None

    return metrics


# ============================================================================
# 平面图评分
# ============================================================================

def score_plan(gt_data: dict, pred_data: dict) -> dict:
    """
    计算平面图子任务分数（G-1, G-2, G-3, C-1, C-2）。

    Args:
        gt_data:   gt["data"] 块
        pred_data: runner 输出 extraction.parsed_output 块

    Returns:
        包含所有指标的字典
    """
    metrics = {}

    gt_grid   = gt_data.get("grid_info", {}) or {}
    pred_grid = pred_data.get("grid_info", {}) or {}

    gt_x   = gt_grid.get("x_axes", []) or []
    gt_y   = gt_grid.get("y_axes", []) or []
    pred_x = pred_grid.get("x_axes", []) or []
    pred_y = pred_grid.get("y_axes", []) or []

    # ---- G-1: 轴网数量 ----
    g1_x_err = abs(len(pred_x) - len(gt_x))
    g1_y_err = abs(len(pred_y) - len(gt_y))
    g1_score = 1.0 if (g1_x_err == 0 and g1_y_err == 0) else 0.0
    metrics["G1_x_error"] = g1_x_err
    metrics["G1_y_error"] = g1_y_err
    metrics["G1_gt_x"]    = len(gt_x)
    metrics["G1_gt_y"]    = len(gt_y)
    metrics["G1_score"]   = g1_score

    # ---- G-2: 轴线标签命中率 ----
    gt_labels   = {str(ax["label"]) for ax in (gt_x + gt_y) if "label" in ax}
    pred_labels = {str(ax["label"]) for ax in (pred_x + pred_y) if "label" in ax}
    g2_hits     = len(gt_labels & pred_labels)
    g2_total    = len(gt_labels)
    metrics["G2_label_hit_rate"] = round(g2_hits / max(g2_total, 1), 4)
    metrics["G2_gt_labels"]      = g2_total
    metrics["G2_hit_labels"]     = g2_hits

    # ---- G-3: 轴网间距 MAE ----
    def compute_spacings(axes: list) -> dict:
        """从轴列表计算相邻间距 {(label_i, label_{i+1}): spacing}"""
        valid = [(str(ax["label"]), float(ax["coordinate"]))
                 for ax in axes if "label" in ax and ax.get("coordinate") is not None]
        valid.sort(key=lambda x: x[1])
        result = {}
        for i in range(len(valid) - 1):
            key = (valid[i][0], valid[i + 1][0])
            result[key] = valid[i + 1][1] - valid[i][1]
        return result

    gt_x_sp   = compute_spacings(gt_x)
    gt_y_sp   = compute_spacings(gt_y)
    pred_x_sp = compute_spacings(pred_x)
    pred_y_sp = compute_spacings(pred_y)

    spacing_errors = []
    spacing_hits   = 0

    for axes_gt_sp, axes_pred_sp in [(gt_x_sp, pred_x_sp), (gt_y_sp, pred_y_sp)]:
        for key, gt_sp in axes_gt_sp.items():
            if key in axes_pred_sp:
                err = abs(axes_pred_sp[key] - gt_sp)
                spacing_errors.append(err)
                if err <= SPACING_TOLERANCE_MM:
                    spacing_hits += 1

    g3_total = len(gt_x_sp) + len(gt_y_sp)
    metrics["G3_spacing_mae"]      = round(sum(spacing_errors) / len(spacing_errors), 2) if spacing_errors else None
    metrics["G3_spacing_hit_rate"] = round(spacing_hits / max(g3_total, 1), 4) if g3_total > 0 else None
    metrics["G3_matched_spacings"] = len(spacing_errors)
    metrics["G3_gt_spacings"]      = g3_total

    # ---- C-1: 柱数量 ----
    gt_components  = gt_data.get("components_above", {}) or {}
    pred_components = pred_data.get("components_above", {}) or {}

    gt_cols   = gt_components.get("columns", []) or []
    pred_cols = pred_components.get("columns", []) or []

    c1_error = abs(len(pred_cols) - len(gt_cols))
    c1_score = max(0.0, 1.0 - c1_error / max(len(gt_cols), 1))
    metrics["C1_count_error"] = c1_error
    metrics["C1_gt_count"]    = len(gt_cols)
    metrics["C1_pred_count"]  = len(pred_cols)
    metrics["C1_score"]       = round(c1_score, 4)

    # ---- C-2: 柱位置 F1（基于 grid_location 精确匹配）----
    # 同一位置可能有多根柱，用 multiset 匹配
    from collections import Counter
    gt_locs   = Counter(
        str(col["grid_location"]) for col in gt_cols
        if col.get("grid_location") is not None
    )
    pred_locs = Counter(
        str(col["grid_location"]) for col in pred_cols
        if col.get("grid_location") is not None
    )

    tp = sum((gt_locs & pred_locs).values())
    fp = sum((pred_locs - gt_locs).values())
    fn = sum((gt_locs - pred_locs).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    metrics["C2_tp"]        = tp
    metrics["C2_fp"]        = fp
    metrics["C2_fn"]        = fn
    metrics["C2_precision"] = round(precision, 4)
    metrics["C2_recall"]    = round(recall, 4)
    metrics["C2_f1"]        = round(f1, 4)

    # 注：梁指标当前禁用（per discussion.md §2.2）
    metrics["_beam_metrics"] = "DISABLED per discussion.md §2.2"

    return metrics


# ============================================================================
# 综合得分计算
# ============================================================================

def compute_weighted_score(drawing_type: str, metrics: dict) -> float:
    """按 discussion.md §5.3 权重计算综合分（0~1）"""
    if drawing_type == "elevation":
        type_score = 1.0 if metrics.get("type_id_correct") else 0.0
        e1    = metrics.get("E1_score", 0.0) or 0.0
        e2    = metrics.get("E2_hit_rate", 0.0) or 0.0
        e3    = metrics.get("E3_floor_height_accuracy", 0.0) or 0.0
        score = (WEIGHTS_ELEVATION["type_id"] * type_score
                 + WEIGHTS_ELEVATION["E1"]     * e1
                 + WEIGHTS_ELEVATION["E2_hit"] * e2
                 + WEIGHTS_ELEVATION["E3"]     * e3)
        return round(score, 4)

    elif drawing_type == "plan":
        type_score = 1.0 if metrics.get("type_id_correct") else 0.0
        g1    = metrics.get("G1_score", 0.0) or 0.0
        g2    = metrics.get("G2_label_hit_rate", 0.0) or 0.0
        g3    = metrics.get("G3_spacing_hit_rate", 0.0) or 0.0
        c1    = metrics.get("C1_score", 0.0) or 0.0
        c2    = metrics.get("C2_f1", 0.0) or 0.0
        score = (WEIGHTS_PLAN["type_id"] * type_score
                 + WEIGHTS_PLAN["G1"]     * g1
                 + WEIGHTS_PLAN["G2"]     * g2
                 + WEIGHTS_PLAN["G3_hit"] * g3
                 + WEIGHTS_PLAN["C1"]     * c1
                 + WEIGHTS_PLAN["C2_f1"]  * c2)
        return round(score, 4)

    return 0.0


# ============================================================================
# 单张图打分
# ============================================================================

def score_image(image_id: str, gt: dict, pred_record: dict) -> dict:
    """
    对单张图纸计算全部评分指标。

    Returns:
        {
          "image_id": ...,
          "gt_drawing_type": ...,
          "pred_drawing_type": ...,
          "type_id_correct": bool,
          "extraction_failed": bool,  # parsed_output 为 None 时为 True
          "metrics": {...},
          "weighted_score": float,
          "errors": [...]
        }
    """
    result = {
        "image_id": image_id,
        "gt_drawing_type": gt.get("drawing_type", "unknown"),
        "pred_drawing_type": None,
        "type_id_correct": False,
        "extraction_failed": False,
        "metrics": {},
        "weighted_score": 0.0,
        "errors": [],
    }

    # 图纸类型识别
    type_id_record = pred_record.get("type_id", {})
    pred_type_parsed = type_id_record.get("parsed_output") or {}
    pred_drawing_type = pred_type_parsed.get("drawing_type", "unknown")
    result["pred_drawing_type"] = pred_drawing_type

    gt_drawing_type = gt.get("drawing_type", "unknown")
    type_correct = (pred_drawing_type == gt_drawing_type)
    result["type_id_correct"] = type_correct
    result["metrics"]["type_id_correct"] = int(type_correct)

    # 提取结果
    ext_record = pred_record.get("extraction", {})
    pred_data  = ext_record.get("parsed_output")
    parse_err  = ext_record.get("parse_error")

    # TRUNCATED_JSON_REPAIRED 表示截断后修复，有部分数据，当作部分成功继续评分
    is_truncated_repair = (parse_err == "TRUNCATED_JSON_REPAIRED")
    if (parse_err and not is_truncated_repair) or pred_data is None:
        result["extraction_failed"] = True
        result["errors"].append(f"提取 JSON 解析失败: {parse_err}")
    if is_truncated_repair:
        result["errors"].append("JSON 截断修复（部分数据）")

    gt_data = gt.get("data", {})
    if not gt_data:
        result["errors"].append("GT data 块为空，跳过子任务评分")
        result["weighted_score"] = compute_weighted_score(gt_drawing_type, result["metrics"])
        return result

    # 根据 GT 图纸类型评分（即使预测类型错误也计算，但综合分中 type_id 为 0）
    try:
        if gt_drawing_type == "elevation":
            sub_metrics = score_elevation(gt_data, pred_data)
        elif gt_drawing_type == "plan":
            sub_metrics = score_plan(gt_data, pred_data)
        else:
            result["errors"].append(f"未知 GT 图纸类型: {gt_drawing_type}")
            sub_metrics = {}
        result["metrics"].update(sub_metrics)
    except Exception as e:
        result["errors"].append(f"评分计算异常: {e}")

    result["weighted_score"] = compute_weighted_score(gt_drawing_type, result["metrics"])
    return result


# ============================================================================
# 汇总统计
# ============================================================================

def aggregate(all_scores: list) -> dict:
    """计算全部图纸的汇总统计"""

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    elevation = [s for s in all_scores if s["gt_drawing_type"] == "elevation"]
    plan      = [s for s in all_scores if s["gt_drawing_type"] == "plan"]
    failed    = [s for s in all_scores if s["extraction_failed"]]

    def agg_group(group, dtype):
        if not group:
            return {}
        keys = {}
        for s in group:
            for k, v in s["metrics"].items():
                if isinstance(v, (int, float)):
                    keys.setdefault(k, []).append(v)
        return {k: mean(v) for k, v in keys.items()}

    return {
        "total_images":    len(all_scores),
        "elevation_count": len(elevation),
        "plan_count":      len(plan),
        "failed_count":    len(failed),
        "type_id_accuracy": mean([s["metrics"].get("type_id_correct") for s in all_scores]),
        "elevation_avg": {
            "weighted_score": mean([s["weighted_score"] for s in elevation]),
            **agg_group(elevation, "elevation"),
        },
        "plan_avg": {
            "weighted_score": mean([s["weighted_score"] for s in plan]),
            **agg_group(plan, "plan"),
        },
        "overall_weighted_score": mean([s["weighted_score"] for s in all_scores]),
    }


# ============================================================================
# 报告生成
# ============================================================================

def write_json_report(scores: list, summary: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image": scores}, f, ensure_ascii=False, indent=2)


def write_markdown_report(scores: list, summary: dict, run_id: str, path: Path):
    lines = [
        f"# Evaluation Report",
        f"",
        f"**Run ID:** `{run_id}`  ",
        f"",
        f"---",
        f"",
        f"## 汇总",
        f"",
        f"| 指标 | 值 |",
        f"|------|----|",
        f"| 总图纸数 | {summary['total_images']} |",
        f"| 立面图 | {summary['elevation_count']} |",
        f"| 平面图 | {summary['plan_count']} |",
        f"| 提取失败 | {summary['failed_count']} |",
        f"| 图纸类型识别准确率 | {_pct(summary.get('type_id_accuracy'))} |",
        f"| 综合加权分（全部） | {_pct(summary.get('overall_weighted_score'))} |",
        f"",
    ]

    # 立面图汇总
    el = summary.get("elevation_avg", {})
    if el:
        lines += [
            f"### 立面图平均指标",
            f"",
            f"| 指标 | 值 |",
            f"|------|----|",
            f"| 综合加权分 | {_pct(el.get('weighted_score'))} |",
            f"| E-1 楼层数得分 | {_pct(el.get('E1_score'))} |",
            f"| E-2 标高命中率（±{ELEVATION_TOLERANCE_MM}mm） | {_pct(el.get('E2_hit_rate'))} |",
            f"| E-2 标高 MAE | {_mm(el.get('E2_elevation_mae'))} |",
            f"| E-3 层高正确率（±{FLOOR_HEIGHT_TOLERANCE}mm） | {_pct(el.get('E3_floor_height_accuracy'))} |",
            f"",
        ]

    # 平面图汇总
    pl = summary.get("plan_avg", {})
    if pl:
        lines += [
            f"### 平面图平均指标",
            f"",
            f"| 指标 | 值 |",
            f"|------|----|",
            f"| 综合加权分 | {_pct(pl.get('weighted_score'))} |",
            f"| G-1 轴网数量得分 | {_pct(pl.get('G1_score'))} |",
            f"| G-2 轴线标签命中率 | {_pct(pl.get('G2_label_hit_rate'))} |",
            f"| G-3 间距命中率（±{SPACING_TOLERANCE_MM}mm） | {_pct(pl.get('G3_spacing_hit_rate'))} |",
            f"| G-3 间距 MAE | {_mm(pl.get('G3_spacing_mae'))} |",
            f"| C-1 柱数量得分 | {_pct(pl.get('C1_score'))} |",
            f"| C-2 柱位置 F1 | {_pct(pl.get('C2_f1'))} |",
            f"| C-2 柱位置 Precision | {_pct(pl.get('C2_precision'))} |",
            f"| C-2 柱位置 Recall | {_pct(pl.get('C2_recall'))} |",
            f"",
        ]

    # 逐图明细
    lines += [
        f"---",
        f"",
        f"## 逐图明细",
        f"",
    ]

    # 立面图明细
    el_scores = [s for s in scores if s["gt_drawing_type"] == "elevation"]
    if el_scores:
        lines += [
            f"### 立面图",
            f"",
            f"| ID | 类型正确 | E1得分 | E2命中率 | E2_MAE | E3层高 | 综合分 | 备注 |",
            f"|----|---------|--------|---------|--------|--------|--------|------|",
        ]
        for s in el_scores:
            m = s["metrics"]
            note = "提取失败" if s["extraction_failed"] else ("; ".join(s["errors"]) if s["errors"] else "")
            lines.append(
                f"| {s['image_id']} "
                f"| {'✓' if s['type_id_correct'] else '✗'} "
                f"| {_pct(m.get('E1_score'))} "
                f"| {_pct(m.get('E2_hit_rate'))} "
                f"| {_mm(m.get('E2_elevation_mae'))} "
                f"| {_pct(m.get('E3_floor_height_accuracy'))} "
                f"| {_pct(s['weighted_score'])} "
                f"| {note} |"
            )
        lines.append("")

    # 平面图明细
    pl_scores = [s for s in scores if s["gt_drawing_type"] == "plan"]
    if pl_scores:
        lines += [
            f"### 平面图",
            f"",
            f"| ID | 类型正确 | G1 | G2命中 | G3间距 | C1柱数 | C2_F1 | 综合分 | 备注 |",
            f"|----|---------|----|----|--------|--------|-------|--------|------|",
        ]
        for s in pl_scores:
            m = s["metrics"]
            note = "提取失败" if s["extraction_failed"] else ("; ".join(s["errors"]) if s["errors"] else "")
            lines.append(
                f"| {s['image_id']} "
                f"| {'✓' if s['type_id_correct'] else '✗'} "
                f"| {_pct(m.get('G1_score'))} "
                f"| {_pct(m.get('G2_label_hit_rate'))} "
                f"| {_pct(m.get('G3_spacing_hit_rate'))} "
                f"| {_pct(m.get('C1_score'))} "
                f"| {_pct(m.get('C2_f1'))} "
                f"| {_pct(s['weighted_score'])} "
                f"| {note} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _pct(v):
    if v is None:
        return "N/A"
    return f"{v * 100:.1f}%"


def _mm(v):
    if v is None:
        return "N/A"
    return f"{v:.1f}mm"


# ============================================================================
# CLI
# ============================================================================

def find_latest_run(out_json_dir: Path) -> Optional[Path]:
    """找 out_json/ 下修改时间最新的 *_result 目录"""
    if not out_json_dir.exists():
        return None
    runs = sorted(
        [d for d in out_json_dir.iterdir()
         if d.is_dir() and d.name.endswith("_result") and (d / "run_config.json").exists()],
        key=lambda d: d.stat().st_mtime,
    )
    return runs[-1] if runs else None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation Scorer（per discussion.md §5）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model",   help="模型目录名，如 gemini-3.1-pro-preview（在 out_json/ 下查找 {model}_result）")
    group.add_argument("--run-dir", help="Runner 输出目录的完整路径")
    group.add_argument("--latest",  action="store_true",
                       help="自动使用 out_json/ 下修改时间最新的模型目录")

    parser.add_argument("--gt-dir",   default="../datasets/ground_truth",
                        help="GT 目录路径（默认 ../datasets/ground_truth）")
    parser.add_argument("--output",   default="../datasets/out_json/reports",
                        help="报告输出目录（默认 ../datasets/out_json/reports）")
    parser.add_argument("--out-json-base", default="../datasets/out_json",
                        help="out_json/ 基础目录（配合 --model / --latest 使用）")
    args = parser.parse_args()

    # 确定 run_dir
    out_json_base = Path(args.out_json_base)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.model:
        run_dir = out_json_base / f"{args.model}_result"
    elif args.latest:
        run_dir = find_latest_run(out_json_base)
        if run_dir is None:
            print(f"在 {out_json_base} 下未找到任何模型结果目录")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    if not run_dir.exists():
        print(f"run 目录不存在：{run_dir}")
        sys.exit(1)

    gt_dir     = Path(args.gt_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_dir.name  # 如 gemini-3.1-pro-preview_result
    print(f"\n{'='*60}")
    print(f"Scorer 运行")
    print(f"模型目录：{run_dir}")
    print(f"GT 目录：  {gt_dir}")
    print(f"{'='*60}\n")

    # 收集所有图纸 ID（从 *_extraction.json 文件名推断，跳过 run_config.json）
    raw_files = sorted(f for f in run_dir.glob("*_extraction.json")
                       if not f.stem.startswith("run_config"))
    if not raw_files:
        print(f"在 {run_dir} 下未找到 *_extraction.json 文件")
        sys.exit(1)

    all_scores = []
    for raw_file in raw_files:
        image_id = raw_file.stem.replace("_extraction", "")
        print(f"  评分 {image_id} ...", end=" ", flush=True)

        gt = load_gt(gt_dir, image_id)
        if gt is None:
            print(f"跳过（GT 不存在或为空）")
            continue

        pred_record = load_prediction(run_dir, image_id)
        if pred_record is None:
            print(f"跳过（预测文件不存在）")
            continue

        score = score_image(image_id, gt, pred_record)
        all_scores.append(score)
        ws = score["weighted_score"]
        flag = "⚠ 提取失败" if score["extraction_failed"] else ""
        print(f"综合分 {ws*100:.1f}%  {flag}")

    if not all_scores:
        print("无有效评分数据，退出")
        sys.exit(1)

    summary = aggregate(all_scores)

    json_out = output_dir / f"{run_id}_scores.json"
    md_out   = output_dir / f"{run_id}_report.md"
    write_json_report(all_scores, summary, json_out)
    write_markdown_report(all_scores, summary, run_id, md_out)

    print(f"\n{'='*60}")
    print(f"综合加权分：{summary['overall_weighted_score'] * 100:.1f}%")
    print(f"  立面图平均：{_pct(summary['elevation_avg'].get('weighted_score'))}")
    print(f"  平面图平均：{_pct(summary['plan_avg'].get('weighted_score'))}")
    print(f"JSON 报告：{json_out}")
    print(f"MD  报告：{md_out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
