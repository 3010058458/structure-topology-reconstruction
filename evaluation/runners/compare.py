"""
compare.py — 全模型横向对比脚本

读取 out_json/ 下所有 *_result 目录（含 skill_result），
与 ground_truth/ 对比评分，输出两个 CSV：

  comparison_summary.csv
      行  = 模型名（skill + baseline 模型）
      列  = 图片 ID（EL-001 … PL-S-009）
      值  = 综合加权分（0~1）

  comparison_detail.csv
      长格式，每行 = (model, image_id, drawing_type, metric, value)
      涵盖所有子指标（E1/E2/E3/G1/G2/G3/C1/C2 及 weighted_score）
      可在 Excel 中通过透视表切换任意视图

用法：
    cd evaluation/runners
    python compare.py
    python compare.py --gt-dir ../datasets/ground_truth \\
                      --out-json ../datasets/out_json \\
                      --output  ../datasets/out_json/reports
"""

import sys
import csv
import json
import argparse
from pathlib import Path

# 复用 scorer.py 中的评分函数
sys.path.insert(0, str(Path(__file__).parent))
from scorer import load_gt, score_image


# ============================================================================
# 数据加载（兼容 skill 格式 和 runner 格式）
# ============================================================================

def _is_skill_dir(result_dir: Path) -> bool:
    """
    判断目录是否为 Skill 输出（skill_result/）。
    Skill 输出文件顶层有 "drawing_type" + "data"，没有 "type_id"/"extraction" 包装。
    """
    return result_dir.name == "skill_result"


def load_pred_record(result_dir: Path, image_id: str, is_skill: bool):
    """
    加载预测文件，统一归一化为 runner 格式供 scorer.score_image() 使用。

    Skill 格式（skill_result/）：
        {drawing_type, floor_id, data, ...}
    Runner 格式（*_result/）：
        {image_id, type_id: {parsed_output: {drawing_type}}, extraction: {parsed_output}}

    Returns None 若文件不存在。
    """
    path = result_dir / f"{image_id}_extraction.json"
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if is_skill:
        # 归一化：把 skill 输出包装成 runner 格式
        drawing_type = raw.get("drawing_type", "unknown")
        data_block   = raw.get("data")
        return {
            "image_id": image_id,
            "type_id": {
                "parsed_output": {"drawing_type": drawing_type},
                "parse_error":   None,
            },
            "extraction": {
                "parsed_output": data_block,
                "parse_error":   None if data_block else "skill data block 为空",
            },
        }
    else:
        return raw


# ============================================================================
# 单模型全量评分
# ============================================================================

def score_model(model_name: str, result_dir: Path, gt_dir: Path) -> dict:
    """
    对 result_dir 下所有图片评分。

    Returns:
        {image_id: score_result}   score_result 结构同 scorer.score_image() 输出
    """
    is_skill = _is_skill_dir(result_dir)
    results  = {}

    for path in sorted(result_dir.glob("*_extraction.json")):
        if path.stem == "run_config":
            continue
        image_id = path.stem.replace("_extraction", "")

        gt = load_gt(gt_dir, image_id)
        if gt is None:
            continue

        pred = load_pred_record(result_dir, image_id, is_skill)
        if pred is None:
            continue

        results[image_id] = score_image(image_id, gt, pred)

    return results


# ============================================================================
# 主程序
# ============================================================================

# 所有可能出现的指标列（固定顺序，缺失时留空）
ELEVATION_METRICS = [
    "weighted_score",
    "type_id_correct",
    "E1_count_error", "E1_score",
    "E2_matched_floors", "E2_gt_floors", "E2_elevation_mae", "E2_hit_rate",
    "E3_floor_height_accuracy",
]
PLAN_METRICS = [
    "weighted_score",
    "type_id_correct",
    "G1_x_error", "G1_y_error", "G1_gt_x", "G1_gt_y", "G1_score",
    "G2_label_hit_rate", "G2_gt_labels", "G2_hit_labels",
    "G3_spacing_mae", "G3_spacing_hit_rate", "G3_matched_spacings", "G3_gt_spacings",
    "C1_count_error", "C1_gt_count", "C1_pred_count", "C1_score",
    "C2_tp", "C2_fp", "C2_fn", "C2_precision", "C2_recall", "C2_f1",
]


def _get_val(score_result: dict, metric: str):
    """从 score_result 中取指标值（weighted_score / type_id_correct 特殊处理）"""
    if metric == "weighted_score":
        return score_result.get("weighted_score")
    if metric == "type_id_correct":
        return int(score_result.get("type_id_correct", 0))
    return score_result.get("metrics", {}).get(metric)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ============================================================================
# 图纸难度权重（影响综合加权均分，不改变单图评分）
# 复杂平面图最难，权重最高；简单平面图和立面图权重基准 1.0
# ============================================================================
def get_image_weight(image_id: str) -> float:
    if image_id.startswith("PL-C"):
        return 3.0   # 复杂平面图：轴网密、构件多，难度最高
    elif image_id.startswith("PL-M"):
        return 1.5   # 中等平面图
    elif image_id.startswith("PL-S"):
        return 1.0   # 简单平面图
    else:
        return 1.0   # 立面图


def weighted_mean(scores_dict: dict, image_ids: list) -> float:
    """按图纸难度权重计算加权均分"""
    total_w, total_ws = 0.0, 0.0
    for img_id in image_ids:
        s = scores_dict.get(img_id)
        if s is None:
            continue
        w = get_image_weight(img_id)
        total_ws += w * s["weighted_score"]
        total_w  += w
    return round(total_ws / total_w, 4) if total_w > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="全模型横向对比（compare.py）")
    parser.add_argument("--gt-dir",   default="../datasets/ground_truth",
                        help="GT 目录（默认 ../datasets/ground_truth）")
    parser.add_argument("--out-json", default="../datasets/out_json",
                        help="模型结果根目录（默认 ../datasets/out_json）")
    parser.add_argument("--output",   default="../datasets/out_json/reports",
                        help="CSV 输出目录（默认 ../datasets/out_json/reports）")
    args = parser.parse_args()

    gt_dir      = Path(args.gt_dir).resolve()
    out_json    = Path(args.out_json).resolve()
    output_dir  = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 收集所有图片 ID（以 GT 为准）────────────────────────────────────
    all_image_ids = sorted(
        p.stem.replace("_extraction", "")
        for p in gt_dir.glob("*_extraction.json")
    )
    if not all_image_ids:
        print(f"GT 目录为空：{gt_dir}")
        return

    # ── 收集所有模型目录（skill 排首位）──────────────────────────────────
    model_dirs = {}   # model_name -> Path
    skill_dir = out_json / "skill_result"
    if skill_dir.exists():
        model_dirs["skill"] = skill_dir

    for d in sorted(out_json.iterdir()):
        if d.is_dir() and d.name.endswith("_result") and d.name != "skill_result":
            model_name = d.name[: -len("_result")]
            model_dirs[model_name] = d

    if not model_dirs:
        print(f"在 {out_json} 下未找到任何 *_result 目录")
        return

    model_names = list(model_dirs.keys())
    print(f"\n找到 {len(model_names)} 个模型：{model_names}")
    print(f"图片数量：{len(all_image_ids)}")

    # ── 评分所有模型 ──────────────────────────────────────────────────────
    all_scores = {}   # {model_name: {image_id: score_result}}
    for model_name, result_dir in model_dirs.items():
        print(f"  评分 [{model_name}] ...", end=" ", flush=True)
        scores = score_model(model_name, result_dir, gt_dir)
        all_scores[model_name] = scores
        print(f"{len(scores)} 张图完成")

    # ── 输出 1：summary CSV（行=模型，列=图片，值=weighted_score）─────────
    summary_path = output_dir / "comparison_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "overall_weighted"] + all_image_ids)
        for model_name in model_names:
            if model_name not in all_scores:
                continue
            overall = weighted_mean(all_scores[model_name], all_image_ids)
            row = [model_name, _fmt(overall)]
            for img_id in all_image_ids:
                s = all_scores[model_name].get(img_id)
                row.append(_fmt(s["weighted_score"]) if s else "")
            writer.writerow(row)
    print(f"\n综合分 CSV   → {summary_path}")

    # ── 输出 2：detail CSV（长格式）────────────────────────────────────────
    # 列：model, image_id, drawing_type, metric, value
    detail_path = output_dir / "comparison_detail.csv"
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "image_id", "drawing_type", "metric", "value"])

        for model_name in model_names:
            if model_name not in all_scores:
                continue
            for img_id in all_image_ids:
                s = all_scores[model_name].get(img_id)
                if s is None:
                    continue
                dt = s.get("gt_drawing_type", "unknown")
                metrics_list = ELEVATION_METRICS if dt == "elevation" else PLAN_METRICS
                for metric in metrics_list:
                    val = _get_val(s, metric)
                    writer.writerow([model_name, img_id, dt, metric, _fmt(val)])

    print(f"详细指标 CSV → {detail_path}")
    print()

    # ── 终端预览综合分 ────────────────────────────────────────────────────
    print("综合加权分预览（难度权重：PL-C×3, PL-M×1.5, EL/PL-S×1）：")
    print(f"  {'模型':<35}", end="")
    for img_id in all_image_ids:
        print(f" {img_id:>10}", end="")
    print()
    for model_name in model_names:
        if model_name not in all_scores:
            continue
        print(f"  {model_name:<35}", end="")
        for img_id in all_image_ids:
            s = all_scores[model_name].get(img_id)
            cell = f"{s['weighted_score']:.2f}" if s else "  -- "
            print(f" {cell:>10}", end="")
        print()

    # ── 难度加权综合排名 ──────────────────────────────────────────────────
    ranking = []
    for model_name in model_names:
        if model_name not in all_scores:
            continue
        scores = all_scores[model_name]
        overall = weighted_mean(scores, all_image_ids)
        el_ids  = [i for i in all_image_ids if i.startswith("EL")]
        plc_ids = [i for i in all_image_ids if i.startswith("PL-C")]
        plm_ids = [i for i in all_image_ids if i.startswith("PL-M")]
        pls_ids = [i for i in all_image_ids if i.startswith("PL-S")]
        ranking.append((model_name, overall,
                        weighted_mean(scores, el_ids),
                        weighted_mean(scores, plc_ids),
                        weighted_mean(scores, plm_ids),
                        weighted_mean(scores, pls_ids)))

    ranking.sort(key=lambda x: -x[1])
    print("\n难度加权综合排名：")
    print("%-4s %-35s %-10s %-10s %-10s %-10s %-10s" % (
        "Rank", "Model", "Overall", "Elevation", "PL-Complex", "PL-Medium", "PL-Simple"))
    print("-" * 85)
    for i, (name, overall, el, plc, plm, pls) in enumerate(ranking, 1):
        flag = " <-- SKILL" if name == "skill" else ""
        print("%-4d %-35s %.4f     %.4f     %.4f     %.4f     %.4f%s" % (
            i, name, overall, el, plc, plm, pls, flag))


if __name__ == "__main__":
    main()
