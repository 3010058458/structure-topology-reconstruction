"""
json_to_vtu.py - 将结构图纸 Skill 输出的 JSON 重建为三维结构模型，导出 VTU 文件

用法:
    python json_to_vtu.py --input ./output --vtu model.vtu
    python json_to_vtu.py --input 1F.json 2F.json elevation.json --vtu model.vtu

输入:
    Skill 输出的 *_extraction.json（立面图 + 一或多张平面图，可混合传入）

输出:
    VTU 文件，可在 ParaView 中查看三维结构模型
    cell_data['component_type']:  0=柱  1=梁  2=剪力墙

坐标系:
    X/Y 来自平面图轴网（单位 mm），Z 来自立面图标高（单位 mm）
    柱/墙竖向：从本楼层楼板面延伸至上一楼层楼板面
    梁水平：位于上一楼层楼板底面（z_top）
"""

import json
import os
import re
import sys
import argparse

import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("[ERROR] 缺少依赖 pyvista")
    print("  安装命令: pip install pyvista")
    sys.exit(1)


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def collect_json_files(paths):
    """从文件路径或目录中收集 *_extraction.json 文件"""
    result = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith("_extraction.json"):
                    result.append(os.path.join(p, name))
        elif os.path.isfile(p) and p.endswith(".json"):
            result.append(p)
    return result


def normalize_floor_id(raw):
    """将各种楼层写法统一为 '1F' 格式，如 '1f' / 'F1' / '一层' 均尝试识别"""
    if not raw:
        return str(raw)
    s = str(raw).strip().upper()
    # F1 -> 1F
    m = re.match(r"^F(\d+)$", s)
    if m:
        return m.group(1) + "F"
    return s


def sort_floor_ids(fids):
    """按楼层编号升序排列，RF / 屋面排到最后"""
    def key(f):
        m = re.match(r"^(\d+)", f)
        return (0, int(m.group(1))) if m else (1, f)
    return sorted(fids, key=key)


# ─────────────────────────────────────────────
# 核心：解析 JSON → 节点 + 单元
# ─────────────────────────────────────────────

def build_model(json_files):
    """
    读取所有 JSON 文件，构建三维结构节点和单元。

    Returns:
        pts_arr     np.ndarray  (N, 3) float64，单位 mm
        cells_arr   np.ndarray  flat VTK cell array
        types_arr   np.ndarray  VTK cell type array
        comp_types  list[int]   0=柱 1=梁 2=墙
    """
    elevation_data = None
    plan_map = {}  # normalized floor_id -> data dict

    # ── 分类读取 JSON ──
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                jdata = json.load(f)
        except Exception as e:
            print(f"  [WARN] 无法读取 {os.path.basename(path)}: {e}")
            continue

        dtype = jdata.get("drawing_type")
        inner = jdata.get("data") or {}

        if dtype == "elevation" and inner.get("floor_levels"):
            elevation_data = inner
            n_levels = len(inner["floor_levels"])
            print(f"  [立面图] {os.path.basename(path)}  楼层数:{n_levels}")

        elif dtype == "plan" and inner.get("components_above"):
            raw_fid = inner.get("floor_id") or jdata.get("floor_id") or "UNK"
            fid = normalize_floor_id(raw_fid)
            plan_map[fid] = inner
            comps = inner["components_above"]
            print(f"  [平面图 {fid}] {os.path.basename(path)}  "
                  f"柱:{len(comps.get('columns', []))} "
                  f"梁:{len(comps.get('beams', []))} "
                  f"墙:{len(comps.get('walls', []))}")

    if not plan_map:
        print("[ERROR] 未找到有效的平面图 JSON（data.components_above 为空）")
        return None

    # ── 建立楼层 Z 坐标映射 ──
    floor_z = {}     # fid -> z_底 (mm)
    floor_ztop = {}  # fid -> z_顶 (mm)

    if elevation_data:
        for lvl in elevation_data["floor_levels"]:
            fid = normalize_floor_id(lvl.get("floor", ""))
            elev = lvl.get("elevation")
            if fid and elev is not None:
                floor_z[fid] = float(elev)
        sorted_fids = sort_floor_ids(list(floor_z.keys()))
        for i, fid in enumerate(sorted_fids):
            if i + 1 < len(sorted_fids):
                floor_ztop[fid] = floor_z[sorted_fids[i + 1]]
            else:
                dz = floor_z[fid] - floor_z[sorted_fids[i - 1]] if i > 0 else 3600.0
                floor_ztop[fid] = floor_z[fid] + dz
        print(f"  [楼层 Z] {dict(sorted(floor_z.items()))}")
    else:
        print("  [WARN] 无立面图，按 3600mm 层高逐层估算 Z 坐标")
        sorted_plans = sort_floor_ids(list(plan_map.keys()))
        for i, fid in enumerate(sorted_plans):
            floor_z[fid] = i * 3600.0
            floor_ztop[fid] = (i + 1) * 3600.0

    # ── 节点去重表 ──
    points = []
    node_idx = {}  # (rx, ry, rz) -> int

    def add_node(x, y, z):
        key = (round(float(x)), round(float(y)), round(float(z)))
        if key not in node_idx:
            node_idx[key] = len(points)
            points.append([float(x), float(y), float(z)])
        return node_idx[key]

    vtk_flat = []   # flat cell array
    vtk_types = []
    comp_types = []

    def push_line(n0, n1, ct):
        if n0 == n1:
            return
        vtk_flat.extend([2, n0, n1])
        vtk_types.append(int(pv.CellType.LINE))
        comp_types.append(ct)

    def push_quad(n0, n1, n2, n3, ct):
        vtk_flat.extend([4, n0, n1, n2, n3])
        vtk_types.append(int(pv.CellType.QUAD))
        comp_types.append(ct)

    # ── 逐楼层生成单元 ──
    for fid, pdata in plan_map.items():
        z_bot = floor_z.get(fid)
        z_top = floor_ztop.get(fid)
        if z_bot is None:
            print(f"  [WARN] 楼层 {fid} 无对应标高，跳过")
            continue
        if z_top is None:
            z_top = z_bot + 3600.0

        comps = pdata.get("components_above", {})

        # 柱：竖向 LINE，从 z_bot 到 z_top
        for col in comps.get("columns", []):
            x, y = col.get("x"), col.get("y")
            if x is None or y is None:
                continue
            push_line(add_node(x, y, z_bot), add_node(x, y, z_top), 0)

        # 梁：水平 LINE，位于 z_top
        for beam in comps.get("beams", []):
            s = beam.get("start")
            e = beam.get("end")
            if not s or not e or len(s) < 2 or len(e) < 2:
                continue
            push_line(add_node(s[0], s[1], z_top), add_node(e[0], e[1], z_top), 1)

        # 剪力墙：四角 QUAD 面单元
        for wall in comps.get("walls", []):
            s = wall.get("start")
            e = wall.get("end")
            if not s or not e or len(s) < 2 or len(e) < 2:
                continue
            n0 = add_node(s[0], s[1], z_bot)
            n1 = add_node(e[0], e[1], z_bot)
            n2 = add_node(e[0], e[1], z_top)
            n3 = add_node(s[0], s[1], z_top)
            push_quad(n0, n1, n2, n3, 2)

    if not points:
        print("[ERROR] 模型为空：没有生成任何节点")
        return None

    return (
        np.array(points, dtype=np.float64),
        np.array(vtk_flat, dtype=np.int64),
        np.array(vtk_types, dtype=np.uint8),
        comp_types,
    )


# ─────────────────────────────────────────────
# 导出 VTU
# ─────────────────────────────────────────────

def export_vtu(pts, cells, types, comp_types, output_path):
    grid = pv.UnstructuredGrid(cells, types, pts)
    grid.cell_data["component_type"] = np.array(comp_types, dtype=np.int32)
    grid.save(output_path)
    return grid


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="将结构图纸 Skill 输出的 JSON 重建为三维 VTU 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  # 处理整个输出目录（自动找所有 extraction.json）\n"
            "  python json_to_vtu.py --input ./output --vtu model.vtu\n\n"
            "  # 指定具体文件\n"
            "  python json_to_vtu.py --input 1F.json 2F.json elevation.json --vtu model.vtu\n"
        ),
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="JSON 文件路径或含 JSON 的目录（可多个）")
    parser.add_argument("--vtu", default="model.vtu",
                        help="输出 VTU 文件路径（默认: model.vtu）")
    args = parser.parse_args()

    print("=" * 60)
    print("结构三维重建  JSON → VTU")
    print("=" * 60)

    # 收集文件
    json_files = collect_json_files(args.input)
    if not json_files:
        print(f"[ERROR] 未找到 JSON 文件：{args.input}")
        sys.exit(1)
    print(f"\n[1/3] 读取 {len(json_files)} 个 JSON 文件...")

    # 构建模型
    print("\n[2/3] 构建三维模型...")
    result = build_model(json_files)
    if result is None:
        sys.exit(1)
    pts, cells, types, comp_types = result

    n_col  = comp_types.count(0)
    n_beam = comp_types.count(1)
    n_wall = comp_types.count(2)
    print(f"  -> 节点: {len(pts)}  单元: {len(comp_types)}"
          f"  （柱:{n_col}  梁:{n_beam}  墙:{n_wall}）")

    # 导出
    print(f"\n[3/3] 导出 → {args.vtu}")
    grid = export_vtu(pts, cells, types, comp_types, args.vtu)
    abs_path = os.path.abspath(args.vtu)
    print(f"  [OK] 已保存: {abs_path}")
    print(f"  [OK] PyVista 网格: {grid.n_points} 个点, {grid.n_cells} 个单元")

    # ParaView 操作说明
    print(f"""
{"=" * 60}
ParaView 查看步骤
{"=" * 60}
1. 下载并安装 ParaView（免费开源）：
   https://www.paraview.org/download/

2. 打开模型文件：
   File → Open → 选择：
   {abs_path}
   → 点击左下角 Apply 按钮

3. 按构件类型着色：
   Properties 面板 → Coloring 下拉框 → 选择 component_type
     0（蓝色）= 柱（Column）
     1（绿色）= 梁（Beam）
     2（红色）= 剪力墙（Wall）

4. 将线单元（柱/梁）显示为圆管：
   Filters 菜单 → Tube
   → 调整 Radius 控制管径（建议先设为跨度的 1/100）

5. 视角操作：
   - 左键拖动：旋转
   - 滚轮：缩放
   - 中键拖动：平移
   - 快捷键 R：重置视角

6. 坐标单位为毫米（mm）
   若需换算为米：Filters → Calculator，表达式填 coords / 1000

{"=" * 60}
""")


if __name__ == "__main__":
    main()
