"""
将 JSONL 标注数据中的 <reason> 内容提取并注入到对应的 parquet 文件中。

数据源: 通过 --jsonl-dir 指定，默认 /data-25T/glx/cot_data/2_images
目标:   支持两种 parquet 目录结构:
  A) {root}/{dataset}/{scene}__hash/{scene}/data/chunk-N/episode.parquet
  B) {root}/{dataset}/{scene}/data/chunk-N/episode.parquet
新增列: 125cm_0deg_reason

用法:
  python inject_reason_to_parquet.py --dry-run
  python inject_reason_to_parquet.py --parquet-root /data-25T/InternData-N1
  python inject_reason_to_parquet.py --parquet-root /data/InternData-N1
  python inject_reason_to_parquet.py --jsonl-dir /path/to/jsonl --parquet-root /path/to/parquet
"""

import json
import re
import glob
import os
import time
import argparse
from collections import defaultdict

import pandas as pd

# ── 常量 ──────────────────────────────────────────────────────────────
REASON_COL = "125cm_0deg_reason"

# 从 image path 提取: dataset, scene_id, chunk, episode_idx, frame_idx
IMG_PATTERN = re.compile(
    r"^(?P<dataset>[^/]+)/(?P<scene>[^/]+)/videos/(?P<chunk>chunk-\d+)/"
    r"observation\.images\.rgb\.\w+/episode_(?P<ep>\d+)_(?P<frame>\d+)\.jpg$"
)
# 提取 reason 标签内的文本
REASON_PATTERN = re.compile(r"<reason>([\s\S]*?)</reason>")


def resolve_parquet_path(parquet_root: str, dataset: str, scene: str,
                         chunk: str, ep: str, cache: dict) -> str | None:
    """
    在 parquet_root 下查找 episode parquet 文件，自动适配两种目录结构:
      A) {root}/{dataset}/{scene}__{hash}/{scene}/data/{chunk}/episode_{ep}.parquet
      B) {root}/{dataset}/{scene}/data/{chunk}/episode_{ep}.parquet
    """
    cache_key = (parquet_root, dataset, scene)
    if cache_key not in cache:
        # 尝试结构 B: 直接 {scene}/data/
        direct = os.path.join(parquet_root, dataset, scene, "data")
        if os.path.isdir(direct):
            cache[cache_key] = ("direct", os.path.join(parquet_root, dataset, scene))
        else:
            # 尝试结构 A: {scene}__*/...
            pattern = os.path.join(parquet_root, dataset, f"{scene}__*")
            matches = glob.glob(pattern)
            if len(matches) >= 1:
                if len(matches) > 1:
                    print(f"[WARN] Multiple dirs for {dataset}/{scene}: {matches}, using first")
                # 结构 A: scene_dir/{scene}/data/...
                scene_dir = matches[0]
                nested = os.path.join(scene_dir, scene, "data")
                if os.path.isdir(nested):
                    cache[cache_key] = ("nested", scene_dir)
                else:
                    cache[cache_key] = None
            else:
                cache[cache_key] = None

    entry = cache[cache_key]
    if entry is None:
        return None

    layout, base_dir = entry
    if layout == "direct":
        # B: {root}/{dataset}/{scene}/data/{chunk}/episode_{ep}.parquet
        return os.path.join(base_dir, "data", chunk, f"episode_{ep}.parquet")
    else:
        # A: {root}/{dataset}/{scene}__hash/{scene}/data/{chunk}/episode_{ep}.parquet
        return os.path.join(base_dir, scene, "data", chunk, f"episode_{ep}.parquet")


def build_reason_index(jsonl_dir: str) -> dict[tuple, dict[int, str]]:
    """
    遍历所有 JSONL 文件，构建 reason 索引。

    返回:
        { (dataset, scene_id, chunk, episode_idx_str): { frame_idx_int: reason_text } }
    """
    index: dict[tuple, dict[int, str]] = defaultdict(dict)
    jsonl_files = sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl")))

    total_lines = 0
    parse_errors = 0
    no_reason = 0

    t0 = time.time()
    for fi, fpath in enumerate(jsonl_files):
        fname = os.path.basename(fpath)
        file_lines = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                file_lines += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue

                # 从最后一张图片路径提取定位信息
                img_path = record["image"][-1]
                m = IMG_PATTERN.match(img_path)
                if not m:
                    parse_errors += 1
                    continue

                dataset = m.group("dataset")
                scene = m.group("scene")
                chunk = m.group("chunk")
                ep = m.group("ep")
                frame = int(m.group("frame"))

                # 提取 reason 文本
                gpt_value = record["conversations"][1]["value"]
                rm = REASON_PATTERN.search(gpt_value)
                if rm:
                    reason_text = rm.group(1).strip()
                    index[(dataset, scene, chunk, ep)][frame] = reason_text
                else:
                    no_reason += 1

        elapsed = time.time() - t0
        print(f"  [{fi+1}/{len(jsonl_files)}] {fname}: {file_lines} lines (total {total_lines}, {elapsed:.0f}s)")

    print(f"\n[Index Summary]")
    print(f"  Total JSONL lines:    {total_lines}")
    print(f"  Parse errors:         {parse_errors}")
    print(f"  Missing <reason>:     {no_reason}")
    print(f"  Unique episodes:      {len(index)}")
    total_frames = sum(len(v) for v in index.values())
    print(f"  Total reason entries: {total_frames}")

    return index


def inject_reasons(
    index: dict[tuple, dict[int, str]],
    parquet_roots: list[str],
    dry_run: bool,
) -> None:
    """将 reason 索引写入对应的 parquet 文件。"""
    total_episodes = len(index)
    success = 0
    skipped_not_found = 0
    frame_mismatch = 0
    total_injected = 0

    # 每个 root 一个缓存
    caches = {root: {} for root in parquet_roots}

    t0 = time.time()

    for i, ((dataset, scene, chunk, ep), frame_reasons) in enumerate(index.items()):
        # 在所有 parquet_roots 中查找
        parquet_path = None
        for root in parquet_roots:
            candidate = resolve_parquet_path(root, dataset, scene, chunk, ep, caches[root])
            if candidate and os.path.exists(candidate):
                parquet_path = candidate
                break

        if parquet_path is None:
            skipped_not_found += 1
            if skipped_not_found <= 10:
                print(f"[SKIP] No parquet found for {dataset}/{scene}/{chunk}/episode_{ep}")
            continue

        # 读取 parquet
        df = pd.read_parquet(parquet_path)

        # 初始化 reason 列
        if REASON_COL not in df.columns:
            df[REASON_COL] = ""

        # 按 frame_index 填入 reason
        injected = 0
        for frame_idx, reason_text in frame_reasons.items():
            mask = df["frame_index"] == frame_idx
            if mask.any():
                df.loc[mask, REASON_COL] = reason_text
                injected += 1
            else:
                frame_mismatch += 1

        total_injected += injected

        # 写回
        if not dry_run:
            df.to_parquet(parquet_path, index=False)

        success += 1

        # 进度
        if (i + 1) % 500 == 0 or (i + 1) == total_episodes:
            elapsed = time.time() - t0
            print(
                f"  Progress: {i+1}/{total_episodes} episodes "
                f"({total_injected} reasons injected, {elapsed:.1f}s)"
            )

    print(f"\n[Inject Summary]")
    print(f"  Parquet roots:          {parquet_roots}")
    print(f"  Total episodes:         {total_episodes}")
    print(f"  Successfully written:   {success}")
    print(f"  Skipped (not found):    {skipped_not_found}")
    print(f"  Frame mismatches:       {frame_mismatch}")
    print(f"  Total reasons injected: {total_injected}")
    if dry_run:
        print(f"  [DRY RUN] No files were modified.")


def verify_sample(
    index: dict[tuple, dict[int, str]],
    parquet_roots: list[str],
    n: int = 5,
) -> None:
    """随机抽样验证注入结果。"""
    import random

    caches = {root: {} for root in parquet_roots}

    keys = list(index.keys())
    random.seed(42)
    samples = random.sample(keys, min(n, len(keys)))

    print(f"\n[Verification] Checking {len(samples)} random episodes...")
    for dataset, scene, chunk, ep in samples:
        parquet_path = None
        for root in parquet_roots:
            candidate = resolve_parquet_path(root, dataset, scene, chunk, ep, caches[root])
            if candidate and os.path.exists(candidate):
                parquet_path = candidate
                break

        if parquet_path is None:
            print(f"  [SKIP] {dataset}/{scene}/ep{ep} - not found in any root")
            continue

        df = pd.read_parquet(parquet_path)
        frame_reasons = index[(dataset, scene, chunk, ep)]
        sample_frame = list(frame_reasons.keys())[0]
        expected = frame_reasons[sample_frame]

        if REASON_COL not in df.columns:
            print(f"  [FAIL] {dataset}/{scene}/ep{ep} - column missing")
            continue

        row = df[df["frame_index"] == sample_frame]
        if row.empty:
            print(f"  [FAIL] {dataset}/{scene}/ep{ep} frame {sample_frame} - row not found")
            continue

        actual = row[REASON_COL].iloc[0]
        match = actual == expected
        status = "OK" if match else "MISMATCH"
        print(
            f"  [{status}] {dataset}/{scene}/ep{ep} frame {sample_frame} "
            f"(in {parquet_path}) "
            f"- reason length: expected={len(expected)}, actual={len(actual)}"
        )
        if not match:
            print(f"    Expected: {expected[:80]}...")
            print(f"    Actual:   {actual[:80]}...")


def main():
    parser = argparse.ArgumentParser(description="Inject reason from JSONL into parquet files")
    parser.add_argument(
        "--jsonl-dir",
        default="/data-25T/glx/cot_data/2_images",
        help="Directory containing JSONL annotation files (default: /data-25T/glx/cot_data/2_images)",
    )
    parser.add_argument(
        "--parquet-root", nargs="+",
        default=["/data-25T/InternData-N1", "/data/InternData-N1"],
        help="One or more parquet root directories to search (default: both /data-25T and /data)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Do not write any files, only print statistics",
    )
    args = parser.parse_args()

    mode = "DRY RUN" if args.dry_run else "WRITE"
    print(f"=== Inject Reason to Parquet ({mode}) ===")
    print(f"  JSONL dir:     {args.jsonl_dir}")
    print(f"  Parquet roots: {args.parquet_root}\n")

    print("[Phase 1] Building reason index from JSONL files...")
    index = build_reason_index(args.jsonl_dir)

    print(f"\n[Phase 2] Injecting reasons into parquet files...")
    inject_reasons(index, args.parquet_root, args.dry_run)

    if not args.dry_run:
        print(f"\n[Phase 3] Verifying...")
        verify_sample(index, args.parquet_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
