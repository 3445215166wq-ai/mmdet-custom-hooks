import argparse
import matplotlib.pyplot as plt
import pandas as pd
import re
import os

# ==================== 消除中文警告 ====================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 命令行参数 ====================
# 用法: python visual.py --log logs/train_20251108_130024.log --out metrics_curve.png
_parser = argparse.ArgumentParser(description='从 mmdet 训练日志里提取 loss / mAP 并画双轴曲线')
_parser.add_argument('--log', default=os.path.join('logs', 'train_20251108_130024.log'),
                     help='mmdet 训练日志 (.log) 路径')
_parser.add_argument('--out', default='metrics_curve.png', help='输出图片路径')
_args = _parser.parse_args()
LOG_PATH = _args.log
OUTPUT_PLOT = _args.out
# ===================================================

# ---------------------- 自动创建输出目录 ----------------------
output_dir = os.path.dirname(OUTPUT_PLOT)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✅ 自动创建输出目录：{output_dir}")

# ---------------------- 读取日志，精准提取指标 ----------------------
epochs = []
train_loss = []
val_mAP = []
val_mAP_50 = []
current_train_epoch = None  # 记录当前训练轮次
current_train_loss = None  # 记录当前轮次的最后一个 step 损失

with open(LOG_PATH, 'r', encoding='gbk') as f:
    for line in f:
        # 1. 匹配训练行：提取轮次和单步 loss（保留最后一个 step 的 loss 作为该轮总损失）
        train_epoch_match = re.search(r'Epoch\(train\) \[(\d+)\]', line)
        loss_match = re.search(r'loss: (\d+\.?\d*)', line)

        if train_epoch_match and loss_match:
            epoch = int(train_epoch_match.group(1))
            step_loss = float(loss_match.group(1))
            # 更新当前轮次和损失（最后一个 step 会覆盖前面的，得到该轮汇总损失）
            current_train_epoch = epoch
            current_train_loss = step_loss

        # 2. 匹配验证行：提取轮次、mAP、mAP50，同时关联当前轮次的训练损失
        val_epoch_match = re.search(r'Epoch\(val\) \[(\d+)\]', line)
        mAP_match = re.search(r'coco/bbox_mAP: (\d+\.?\d*)', line)
        mAP50_match = re.search(r'coco/bbox_mAP_50: (\d+\.?\d*)', line)

        if val_epoch_match and mAP_match and mAP50_match:
            val_epoch = int(val_epoch_match.group(1))
            # 验证轮次和训练轮次一致时，才添加数据（确保 loss 和轮次对应）
            if val_epoch == current_train_epoch and current_train_loss is not None:
                epochs.append(val_epoch)
                val_mAP.append(float(mAP_match.group(1)))
                val_mAP_50.append(float(mAP50_match.group(1)))
                train_loss.append(current_train_loss)
                # 重置，避免重复添加
                current_train_epoch = None
                current_train_loss = None

# ---------------------- 检查数据 ----------------------
if len(epochs) == 0:
    print("⚠️  没提取到数据！检查日志路径是否正确")
    exit()
else:
    print(f"✅ 提取到 {len(epochs)} 轮数据！")
    print(f"Epochs: {epochs}")
    print(f"Training Loss: {train_loss}")  # 打印损失，确认提取成功
    print(f"mAP@50: {val_mAP_50}")

# ---------------------- 生成可视化图 ----------------------
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制训练损失曲线（现在会显示真实变化）
ax1.plot(epochs, train_loss, 'r-', label='Training Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', color='r', fontsize=12)
ax1.tick_params(axis='y', labelcolor='r')
ax1.grid(alpha=0.3)

# 绘制 mAP 曲线
ax2 = ax1.twinx()
ax2.plot(epochs, val_mAP, 'b-', label='mAP', linewidth=2)
ax2.plot(epochs, val_mAP_50, 'g--', label='mAP@50', linewidth=2)
ax2.set_ylabel('mAP', color='b', fontsize=12)
ax2.tick_params(axis='y', labelcolor='b')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

# 标题
plt.title('Training Loss & Validation mAP Curve', fontsize=14)

# 保存图片
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"✅ 指标图已保存到：{OUTPUT_PLOT}")