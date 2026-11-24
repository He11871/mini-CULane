from pathlib import Path
import shutil

SRC_ROOT = Path(r"F:/dataset/CULane")
DST_ROOT = Path(r"F:/dataset/mini_culane")

def ensure_lines_for(dst_jpg: Path):
    """确保 dst 的 *.jpg 旁边有 *.lines.txt；若缺就从 SRC 复制（SRC 可能是 .lines 或 .lines.txt）"""
    dst_lines_txt = dst_jpg.with_suffix(".lines.txt")
    if dst_lines_txt.exists():
        return "ok"
    # 在 SRC 中找到对应的 jpg
    rel = dst_jpg.relative_to(DST_ROOT)
    src_jpg = (SRC_ROOT / rel)
    src_lines_txt = src_jpg.with_suffix(".lines.txt")
    src_lines     = src_jpg.with_suffix(".lines")
    src = None
    if src_lines_txt.exists():
        src = src_lines_txt
    elif src_lines.exists():
        src = src_lines
    else:
        return "src-missing"  # 源端也没有标注
    dst_lines_txt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_lines_txt)
    return "copied"

if __name__ == "__main__":
    total = copied = missing = 0
    for jpg in DST_ROOT.rglob("*.jpg"):
        total += 1
        tag = ensure_lines_for(jpg)
        if tag == "copied": copied += 1
        elif tag == "src-missing": missing += 1
    print(f"[walk] jpg={total}, copied_lines={copied}, src_missing={missing}")
