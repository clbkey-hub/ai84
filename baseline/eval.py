# -*- coding: utf-8 -*-
"""评估脚本：按官方规则计算 5个单选字段Accuracy + 3个多选字段Micro-F1
用法: python eval.py --pred <预测jsonl> --gt <标注jsonl>
"""
import argparse, json, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import SINGLE_FIELDS, MULTI_FIELDS


def load(path, key="video_file"):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            d[r[key]] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()

    pred, gt = load(args.pred), load(args.gt)
    keys = sorted(set(pred) & set(gt))
    print(f"参与评估样本数: {len(keys)} (pred={len(pred)}, gt={len(gt)})\n")

    scores = {}
    for f in SINGLE_FIELDS:
        correct = sum(1 for k in keys if pred[k][f] == gt[k][f])
        scores[f] = correct / len(keys)
        print(f"[单选] {f:22s} Acc = {scores[f]:.4f} ({correct}/{len(keys)})")

    for f in MULTI_FIELDS:
        tp = fp = fn = 0
        for k in keys:
            p, g = set(pred[k][f]), set(gt[k][f])
            tp += len(p & g); fp += len(p - g); fn += len(g - p)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores[f] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"[多选] {f:22s} Micro-F1 = {scores[f]:.4f} (P={prec:.3f} R={rec:.3f})")

    avg = sum(scores.values()) / 8
    print(f"\nOverall Avg = {avg:.4f}   预估初赛得分 = {avg*70:.2f} / 70")


if __name__ == "__main__":
    main()
