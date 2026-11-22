"""
把指定验证图的真值框（绿）与预测框（红，score>=阈值）叠在一起存成 PNG，
用来人眼核对"被判为误检的框到底框在哪"。

用途：mAP 这类指标只告数、不告形状。当怀疑标注不全或有系统性误检时，
看几张最差图比看汇总数字有效得多——本项目就是靠这一步发现
"背景误检"里混着大量"检测正确但未标注"的框，进而推翻了原先的结论。

用法：
    python inspect_predictions.py \
        --pred results/val_predictions.coco.json \
        --gt   /path/to/fs_coco/annotations/val_annotations.json \
        --img-dir /path/to/fs_coco/val \
        --out-dir viz --thr 0.5 \
        smoke_000029.jpg fire_000963.jpg
"""
import argparse
import json
import os

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser(description='叠加真值与预测框，人工核对误检')
    ap.add_argument('--pred', default='results/val_predictions.coco.json')
    ap.add_argument('--gt', required=True, help='COCO 标注文件')
    ap.add_argument('--img-dir', required=True, help='图片目录')
    ap.add_argument('--out-dir', default='viz')
    ap.add_argument('--thr', type=float, default=0.5, help='只画 score>=thr 的预测框')
    ap.add_argument('names', nargs='+', help='要检查的文件名（可多个）')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.pred, encoding='utf-8') as f:
        preds = json.load(f)
    with open(args.gt, encoding='utf-8') as f:
        gt = json.load(f)

    name2id = {i['file_name']: i['id'] for i in gt['images']}
    gt_by_img, pred_by_img = {}, {}
    for a in gt['annotations']:
        gt_by_img.setdefault(a['image_id'], []).append(a)
    for p in preds:
        pred_by_img.setdefault(p['image_id'], []).append(p)

    for name in args.names:
        img_id = name2id.get(name)
        if img_id is None:
            print(f'  跳过（不在标注里）: {name}')
            continue
        path = os.path.join(args.img_dir, name)
        if not os.path.exists(path):
            print(f'  找不到图片: {path}')
            continue

        im = Image.open(path).convert('RGB')
        dr = ImageDraw.Draw(im)
        for a in gt_by_img.get(img_id, []):
            x, y, w, h = a['bbox']
            dr.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)   # 真值绿
        shown = 0
        for p in pred_by_img.get(img_id, []):
            if p['score'] < args.thr:
                continue
            x, y, w, h = p['bbox']
            dr.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)   # 预测红
            dr.text((x + 2, max(0, y - 12)), f"{p['score']:.2f}", fill=(255, 255, 0))
            shown += 1

        out = os.path.join(args.out_dir, name.rsplit('.', 1)[0] + '_viz.png')
        im.save(out)
        print(f'  {name}: 真值 {len(gt_by_img.get(img_id, []))} 框 / '
              f'预测(>= {args.thr}) {shown} 框 -> {out}')


if __name__ == '__main__':
    main()
