#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 mini_culane 的完整性与对齐性：
1) train_gt.txt / val_gt.txt:
   - 检查 <img_rel> <lbl_rel> 1 1 1 1 格式
   - 检查 mini 下 img / lbl / *.lines.txt 是否存在
   - 对比原始 CULane 对应文件是否存在；默认比对文件大小(快)，--checksum 时比 MD5(慢但更稳)
2) test.txt / test_split/*.txt:
   - 检查所有图片存在
   - 检查 test_split 之和 ⊆ test.txt
   - 统计 9 场景样本数，是否均衡
3) 统计各 driver_* 分布；打印前若干错误样例
4) 可选 --vis N: 抽样 N 条生成可视化到 out_dir（叠加 seg mask 与 .lines.txt 折线）

用法示例：
python verify_miniculane.py --src_root F:/dataset/CULane --dst_root F:/dataset/mini_culane --checksum --vis 12 --out_dir verify_vis
"""

import argparse, os, sys, hashlib, random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

SCENE_ALIASES = {
    'normal': ('normal.txt','test0_normal.txt'),
    'crowded': ('crowded.txt','test1_crowd.txt','test1_crowded.txt'),
    'hlight': ('dazzle_light.txt','test2_hlight.txt','test2_dazzle_light.txt'),
    'shadow': ('shadow.txt','test3_shadow.txt'),
    'noline': ('no_line.txt','test4_noline.txt','test4_no_line.txt'),
    'arrow':  ('arrow.txt','test5_arrow.txt'),
    'curve':  ('curve.txt','test6_curve.txt'),
    'cross':  ('cross.txt','test7_cross.txt','test7_crossroad.txt'),
    'night':  ('night.txt','test8_night.txt'),
}

def md5sum(p, blk=1<<20):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        while True:
            b = f.read(blk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def read_lines(p):
    return [ln.strip() for ln in open(p,'r',encoding='utf-8',errors='ignore') if ln.strip()]

def parse_train_line(line):
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    img_rel = parts[0].lstrip('/').replace('\\','/')
    lbl_rel = parts[1].lstrip('/').replace('\\','/')
    extras  = parts[2:] if len(parts) > 2 else ['1','1','1','1']
    # 规范 extras 长度为 4（有些数据可能缺失，补齐）
    if len(extras) < 4:
        extras = extras + ['1']*(4-len(extras))
    return img_rel, lbl_rel, extras

def driver_of(rel_path:str)->str:
    return rel_path.split('/')[0]

def check_pair(img_rel, lbl_rel, src_root:Path, dst_root:Path, do_md5=False):
    """返回 (ok, err_msg, detail_dict)"""
    detail = {}
    dst_img = dst_root / img_rel
    dst_lbl = dst_root / lbl_rel
    dst_lin = dst_img.with_suffix('.lines.txt')

    src_img = src_root / img_rel
    src_lbl = src_root / lbl_rel
    src_lin = src_img.with_suffix('.lines.txt')

    # 1) 存在性
    miss = []
    for p, name in [(dst_img,'dst_img'),(dst_lbl,'dst_lbl'),(dst_lin,'dst_lines'),
                    (src_img,'src_img'),(src_lbl,'src_lbl'),(src_lin,'src_lines')]:
        detail[name] = str(p)
        if not p.exists():
            miss.append(name)
    if miss:
        return False, f"missing: {', '.join(miss)}", detail

    # 2) 大小一致（快速）/ MD5 一致（慢）
    def same(a:Path,b:Path):
        if not do_md5:
            return a.stat().st_size == b.stat().st_size
        return md5sum(a)==md5sum(b)

    size_err = []
    if not same(dst_img, src_img): size_err.append('img')
    if not same(dst_lbl, src_lbl): size_err.append('lbl')
    if not same(dst_lin, src_lin): size_err.append('lines')
    if size_err:
        return False, f"content_diff: {','.join(size_err)}", detail

    # 3) 标签路径一致性（推荐但不强制）：lbl_rel 应该以 'laneseg_label_w16/' + img_rel 路径为后缀
    suffix_ok = lbl_rel.endswith(img_rel.replace('.jpg','.png'))
    if not suffix_ok and 'laneseg_label_w16' in lbl_rel:
        return False, "lbl_rel not aligned with img_rel", detail

    return True, "", detail

def draw_lines_on_image(img, lines_txt_path:Path, color=(0,255,0), thickness=2):
    try:
        pts = []
        with open(lines_txt_path,'r') as f:
            for ln in f:
                vals = ln.strip().split()
                if len(vals) >= 4:
                    xy = list(map(float, vals))
                    pts.append([(int(xy[i]), int(xy[i+1])) for i in range(0,len(xy),2)])
        for poly in pts:
            for i in range(1, len(poly)):
                cv2.line(img, poly[i-1], poly[i], color, thickness, lineType=cv2.LINE_AA)
    except Exception:
        pass
    return img

def overlay_mask(img, mask_path:Path, alpha=0.35):
    if not mask_path.exists(): return img
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None: return img
    m3 = cv2.merge([m,m,m])
    # 简单将车道像素着色为蓝色
    color = np.zeros_like(img)
    color[:,:,2] = (m>0).astype(np.uint8)*255
    out = cv2.addWeighted(img, 1.0, color, alpha, 0)
    return out

def verify(src_root, dst_root, checksum=False, vis=0, out_dir='verify_vis', max_err_show=20):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    errs = []
    counts = dict(train=0, val=0, test=0)
    driver_cnt = dict(train=defaultdict(int), val=defaultdict(int))
    scene_cnt  = defaultdict(int)

    # --- train / val ---
    for split in ['train_gt.txt','val_gt.txt']:
        list_path = dst_root/'list'/split
        if not list_path.exists():
            errs.append(f"[{split}] not found: {list_path}")
            continue
        rows = read_lines(list_path)
        for ln in rows:
            parsed = parse_train_line(ln)
            if not parsed:
                errs.append(f"[{split}] bad line: {ln}")
                continue
            img_rel, lbl_rel, extras = parsed
            ok, msg, detail = check_pair(img_rel, lbl_rel, src_root, dst_root, do_md5=checksum)
            if not ok:
                errs.append(f"[{split}] {msg} :: {ln}")
            counts['train' if split=='train_gt.txt' else 'val'] += 1
            driver_cnt['train' if split=='train_gt.txt' else 'val'][driver_of(img_rel)] += 1

    # --- test / test_split ---
    test_list = dst_root/'list'/'test.txt'
    if not test_list.exists():
        errs.append(f"[test] list not found: {test_list}")
    else:
        test_imgs = [ln.lstrip('/').replace('\\','/') for ln in read_lines(test_list)]
        for rel in test_imgs:
            p = dst_root/rel
            if not p.exists():
                errs.append(f"[test] missing image: {rel}")
        counts['test'] = len(test_imgs)

        # scenes
        # scenes（支持多命名别名）
        sp_dir = dst_root / 'list' / 'test_split'
        missing_scene = []
        for scene, aliases in SCENE_ALIASES.items():
            file_found = None
            for fn in aliases:
                p = sp_dir / fn
                if p.exists():
                    file_found = fn
                    break
            if not file_found:
                missing_scene.append(aliases[0])
                continue

            # 正常读取
            items = [ln.lstrip('/').replace('\\', '/') for ln in read_lines(sp_dir / file_found)]
            for it in items:
                scene_cnt[scene] += 1
            notin = [it for it in items if it not in test_imgs]
            if notin:
                errs.append(f"[test_split:{file_found}] {len(notin)} items not in test.txt (e.g., {notin[:3]})")

        if missing_scene:
            errs.append(f"[test_split] missing files: {', '.join(missing_scene)}")

    # --- 统计与均衡性提示 ---
    print("\n=== Counts ===")
    print(f"Train: {counts['train']}  Val: {counts['val']}  Test: {counts['test']}")
    print("\nDriver distribution (train):")
    for d,c in sorted(driver_cnt['train'].items()): print(f"  {d}: {c}")
    print("Driver distribution (val):")
    for d,c in sorted(driver_cnt['val'].items()): print(f"  {d}: {c}")
    if scene_cnt:
        print("\nScene distribution (test_split):")
        for scene in SCENE_ALIASES.keys():
            print(f"  {scene:>12}: {scene_cnt[scene]}")

    # --- 错误汇总 ---
    if errs:
        print("\n=== ERRORS (showing first {} of {}): ===".format(min(max_err_show,len(errs)), len(errs)))
        for e in errs[:max_err_show]:
            print(" -", e)
    else:
        print("\nAll checks passed ✅")

    # --- 可视化抽样 ---
    if vis > 0:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pool = []
        for split in ['train_gt.txt','val_gt.txt']:
            lp = dst_root/'list'/split
            if lp.exists():
                pool.extend(read_lines(lp))
        random.shuffle(pool)
        take = pool[:min(vis,len(pool))]
        print(f"\nSaving {len(take)} visual samples to: {out_dir.resolve()}")
        for i, ln in enumerate(take):
            img_rel, lbl_rel, _ = parse_train_line(ln)
            img_p = dst_root / img_rel
            lbl_p = dst_root / lbl_rel
            lin_p = img_p.with_suffix('.lines.txt')
            img  = cv2.imread(str(img_p))
            if img is None: continue
            vis_img = overlay_mask(img, lbl_p, alpha=0.35)
            vis_img = draw_lines_on_image(vis_img, lin_p, color=(0,255,0), thickness=2)
            cv2.imwrite(str(out_dir/f"sample_{i:02d}.jpg"), vis_img)

    # 返回非零退出码以便脚本/CI 检测
    return 1 if errs else 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', default='F:\\dataset\\CULane', help='原始 CULane 根目录')
    ap.add_argument('--dst_root', default='F:\\dataset\\mini_culane', help='mini_culane 根目录')
    ap.add_argument('--checksum', action='store_true', help='开启 MD5 校验（更慢但更稳）')
    ap.add_argument('--vis', type=int, default=0, help='可视化抽样数量，0 表示关闭')
    ap.add_argument('--out_dir', default='verify_vis', help='可视化输出目录')
    args = ap.parse_args()
    sys.exit( verify(args.src_root, args.dst_root, checksum=args.checksum, vis=args.vis, out_dir=args.out_dir) )
