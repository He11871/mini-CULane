#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_mini_culane.py
从原始 CULane 数据集中构建一个体量更小但结构兼容的 mini 版本：
- 复制抽样后的图像与对应标签到 <dst_root>（默认: mini_culane）
- 重新生成 list/train_gt.txt、val_gt.txt、test.txt
- 测试集严格按 test_split 的 9 个场景均衡抽样
- 训练/验证集按 driver_* 目录均衡抽样（train/val 没有官方场景标签）

用法示例：
python make_mini_culane.py --src_root F:/dataset/CULane --dst_root F:/dataset/mini_culane ^
    --train_n 10000 --val_n 1000 --test_per_scene 300 --seed 42
"""

import argparse, os, random, shutil
from collections import defaultdict
from pathlib import Path

SCENE_FILES = [
    'test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt',
    'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt'
]

def read_lines(p):
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        return [ln.strip() for ln in f if ln.strip()]

def write_lines(p, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

def rel_img_and_lbl_from_train_gt_line(line: str):
    """
    支持以下两种格式：
    1) <img_rel> <lbl_rel>
    2) <img_rel> <lbl_rel> <exist1> <exist2> <exist3> <exist4>
    同时自动去掉行首的 '/' 并统一成 POSIX 分隔符，便于跨平台拷贝。
    """
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Bad train_gt line (need >=2 columns): {line}")
    img_rel, lbl_rel = parts[0], parts[1]
    # 兼容以 '/' 起始的写法与 Windows 反斜杠
    img_rel = img_rel.lstrip('/').replace('\\', '/')
    lbl_rel = lbl_rel.lstrip('/').replace('\\', '/')
    return img_rel, lbl_rel


def rel_img_from_plain_line(line: str):
    """用于 train/val/test.txt 只有图片路径的情况，也兼容以 '/' 起始。"""
    return line.strip().lstrip('/').replace('\\', '/')


def group_by_driver(paths):
    """按 driver_* 顶级目录分组，便于均衡抽样"""
    g = defaultdict(list)
    for p in paths:
        top = p.split('/')[0]
        g[top].append(p)
    return g

def sample_balanced_by_driver(items, k, seed=42):
    """items 为 <img_rel_path> 或 (img_rel, lbl_rel)；按 driver 均衡抽样到 k 个"""
    random.Random(seed).shuffle(items)
    # 先分组
    def topdir(x):
        p = x[0] if isinstance(x, tuple) else x
        return p.split('/')[0]
    groups = defaultdict(list)
    for it in items:
        groups[topdir(it)].append(it)
    drivers = sorted(groups.keys())
    per = max(1, k // max(1, len(drivers)))
    out = []
    # 轮转抽取，尽量均衡
    idx = {d:0 for d in drivers}
    while len(out) < k:
        progressed = False
        for d in drivers:
            if idx[d] < len(groups[d]):
                out.append(groups[d][idx[d]])
                idx[d] += 1
                progressed = True
                if len(out) >= k: break
        if not progressed:  # 不足时直接结束
            break
    return out[:k]

def copy_img_and_label(src_root, dst_root, img_rel, lbl_rel=None):
    src_img = Path(src_root) / img_rel
    dst_img = Path(dst_root) / img_rel
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    if lbl_rel:
        src_lbl = Path(src_root) / lbl_rel
        dst_lbl = Path(dst_root) / lbl_rel
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_lbl, dst_lbl)

def build_test_from_splits(src_root, per_scene=300, seed=42):
    """从 test_split/ 九个场景各抽 per_scene，返回 test 列表（仅图像路径）"""
    rnd = random.Random(seed)
    test_split_dir = Path(src_root) / 'list' / 'test_split'
    all_rows = []
    for fn in SCENE_FILES:
        lst = read_lines(test_split_dir / fn)
        rnd.shuffle(lst)
        take = lst[:per_scene] if per_scene > 0 else lst
        all_rows.extend(take)
    return all_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', default='F:\\dataset\\CULane', help='原始 CULane 根目录')
    ap.add_argument('--dst_root', default='F:\\dataset\\mini_culane', help='输出根目录（将创建兼容结构）')
    ap.add_argument('--train_n', type=int, default=10000, help='训练集抽样数量（train_gt）')
    ap.add_argument('--val_n', type=int, default=1000, help='验证集抽样数量（val_gt）')
    ap.add_argument('--test_per_scene', type=int, default=300, help='每个场景抽样张数（test_split）')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)

    src = Path(args.src_root)
    dst = Path(args.dst_root)

    # 基本检查
    need = [
        src/'list'/'train_gt.txt',
        src/'list'/'val_gt.txt',
        src/'list'/'test.txt',
        src/'laneseg_label_w16',
    ] + [src/'list'/'test_split'/fn for fn in SCENE_FILES]
    for p in need:
        if not p.exists():
            raise FileNotFoundError(f'缺少必要文件/目录: {p}')

    # 1) 读取 train_gt / val_gt
    train_pairs = []
    for ln in read_lines(src/'list'/'train_gt.txt'):
        img_rel, lbl_rel = rel_img_and_lbl_from_train_gt_line(ln)
        train_pairs.append((img_rel, lbl_rel))

    val_pairs = []
    for ln in read_lines(src/'list'/'val_gt.txt'):
        img_rel, lbl_rel = rel_img_and_lbl_from_train_gt_line(ln)
        val_pairs.append((img_rel, lbl_rel))

    # 2) 训练/验证集：按 driver_* 均衡抽样
    train_pairs = sample_balanced_by_driver(train_pairs, k=min(args.train_n, len(train_pairs)), seed=args.seed)
    val_pairs = sample_balanced_by_driver(val_pairs, k=min(args.val_n, len(val_pairs)), seed=args.seed)

    # 3) 测试集：严格按九场景均衡抽样（仅图像路径）
    test_imgs = build_test_from_splits(src, per_scene=args.test_per_scene, seed=args.seed)

    # 4) 复制文件到 dst_root
    # 训练/验证：复制图片和标签
    for img_rel, lbl_rel in train_pairs:
        copy_img_and_label(src, dst, img_rel, lbl_rel)
    for img_rel, lbl_rel in val_pairs:
        copy_img_and_label(src, dst, img_rel, lbl_rel)

    # 测试：复制图片（标签在官方 Test 不公开，这里只复制图片；若你希望评估，也可从 val_gt 选一份作为 mini-test 自评）
    for img_rel in test_imgs:
        copy_img_and_label(src, dst, img_rel, lbl_rel=None)

    # 5) 生成 mini 的 list 文件
    # train_gt.txt / val_gt.txt：保持“img_rel label_rel”格式
    train_list_lines = [f'{img} {lbl}' for img, lbl in train_pairs]
    val_list_lines   = [f'{img} {lbl}' for img, lbl in val_pairs]
    # test.txt：仅图像路径
    test_list_lines  = test_imgs

    write_lines(dst/'list'/'train_gt.txt', train_list_lines)
    write_lines(dst/'list'/'val_gt.txt',   val_list_lines)
    write_lines(dst/'list'/'test.txt',     test_list_lines)

    # 同时拷贝 test_split 的“子集定义”，便于分场景统计（可选：用我们抽到的 test_imgs 重新生成）
    # 这里我们把 test_split 也重建为“与抽样一致”的版本
    sub_splits = defaultdict(list)
    test_split_dir = src/'list'/'test_split'
    for fn in SCENE_FILES:
        paths = read_lines(test_split_dir/fn)
        picked = set(test_imgs) & set(paths)
        sub_splits[fn] = sorted(picked)
        write_lines(dst/'list'/'test_split'/fn, sub_splits[fn])

    print('Done.')
    print(f'Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_imgs)}')
    print(f'Mini CULane root: {dst.resolve()}')
    print('现在你可以把 data_root 改为 mini_culane 后，直接用原训练/测试脚本运行。')

if __name__ == '__main__':
    main()
