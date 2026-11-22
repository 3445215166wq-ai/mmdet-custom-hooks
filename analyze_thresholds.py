"""
按置信度阈值拆开看这套检测器的真实能力边界。

mAP 是跨所有阈值平均后的单个数，会掩盖"某个阈值区间其实可用"这件事。
本脚本把 COCO 格式的预测与真值按阈值重新匹配（同类别、IoU>=0.5、按 score 降序贪心），
输出每个阈值下的 TP / FP / FN / 精确率 / 召回率 / F1，分别给出全类别与按类别两种口径，
并额外统计图片级命中率（有真值的图里至少被正确框中一个的比例）。

用法：
    python analyze_thresholds.py \
        --pred results/val_predictions.coco.json \
        --gt   E:/dataset/fs_coco/annotations/val_annotations.json
"""
import argparse
import json
from collections import defaultdict

IOU_THR = 0.5
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
              0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]


def iou(a, b):
    """COCO 格式 bbox: [x, y, w, h]"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description='按置信度阈值评估检测结果')
    ap.add_argument('--pred', default='results/val_predictions.coco.json')
    ap.add_argument('--gt', required=True, help='COCO 标注文件（val）')
    ap.add_argument('--iou', type=float, default=IOU_THR)
    args = ap.parse_args()

    with open(args.pred, encoding='utf-8') as f:
        preds = json.load(f)
    with open(args.gt, encoding='utf-8') as f:
        gt = json.load(f)

    gt_by_img = defaultdict(list)
    for a in gt['annotations']:
        gt_by_img[a['image_id']].append(a)
    pred_by_img = defaultdict(list)
    for p in preds:
        pred_by_img[p['image_id']].append(p)

    cats = {c['id']: c['name'] for c in gt.get('categories', [])}
    gt_per_cat = defaultdict(int)
    for a in gt['annotations']:
        gt_per_cat[a['category_id']] += 1

    print(f"验证集：{len(gt['images'])} 张图 / {len(gt['annotations'])} 个真值框 / {len(preds)} 个预测框")
    print(f"平均每图预测 {len(preds)/len(gt['images']):.1f} 框，"
          f"真值 {len(gt['annotations'])/len(gt['images']):.2f} 框\n")

    def match(thr, cat_id=None):
        tp = fp = fn = 0
        for img_id, gts in gt_by_img.items():
            gts = [g for g in gts if cat_id is None or g['category_id'] == cat_id]
            ps = [p for p in pred_by_img.get(img_id, [])
                  if p['score'] >= thr and (cat_id is None or p['category_id'] == cat_id)]
            ps.sort(key=lambda x: -x['score'])
            matched = [False] * len(gts)
            for p in ps:
                best_i, best_iou = -1, args.iou
                for i, g in enumerate(gts):
                    if matched[i]:
                        continue
                    v = iou(p['bbox'], g['bbox'])
                    if v >= best_iou:
                        best_i, best_iou = i, v
                if best_i >= 0:
                    matched[best_i] = True
                    tp += 1
                else:
                    fp += 1
            fn += matched.count(False)
        return tp, fp, fn

    def report(cat_id=None, title='全部类别'):
        print(f"=== {title} ===")
        print(f"{'阈值':>6}{'TP':>7}{'FP':>7}{'FN':>7}{'精确率':>9}{'召回率':>9}{'F1':>8}")
        best = None
        for t in THRESHOLDS:
            tp, fp, fn = match(t, cat_id)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            print(f"{t:>6.2f}{tp:>7}{fp:>7}{fn:>7}{prec:>9.3f}{rec:>9.3f}{f1:>8.3f}")
            if best is None or f1 > best[1]:
                best = (t, f1, prec, rec)
        print(f"  → F1 最优阈值 {best[0]:.2f}：P={best[2]:.3f} R={best[3]:.3f} F1={best[1]:.3f}\n")

    report(None, '全部类别（fire + smoke）')
    for cid, name in cats.items():
        report(cid, f'仅 {name}（真值 {gt_per_cat[cid]} 框）')

    print('=== 图片级：有真值的图里至少有一个框命中 ===')
    print(f"{'阈值':>6}{'有真值图':>10}{'至少命中1个':>12}{'占比':>8}")
    for t in [0.10, 0.20, 0.30, 0.50]:
        hit = tot = 0
        for img_id, gts in gt_by_img.items():
            if not gts:
                continue
            tot += 1
            ok = False
            for p in pred_by_img.get(img_id, []):
                if p['score'] < t:
                    continue
                for g in gts:
                    if p['category_id'] == g['category_id'] and iou(p['bbox'], g['bbox']) >= args.iou:
                        ok = True
                        break
                if ok:
                    break
            hit += ok
        print(f"{t:>6.2f}{tot:>10}{hit:>12}{hit/tot:>8.3f}")


if __name__ == '__main__':
    main()
