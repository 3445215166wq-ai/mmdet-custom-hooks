from mmengine import Config
from mmengine.runner import Runner
from mmengine.registry import VISUALIZERS, HOOKS
from mmdet.visualization import DetLocalVisualizer
from mmengine.hooks import Hook
from mmdet.datasets import CocoDataset
from mmdet.models import FasterRCNN
from mmdet.utils import register_all_modules
import os
import torch
# 新增：用于绘制收敛曲线
import matplotlib.pyplot as plt
import numpy as np

# ----------------=== 关键：注册MMDetection所有模块（避免未注册报错）===----------------
register_all_modules()

# ----------------=== 修复+简化：训练可视化Hook（彻底解决格式匹配问题）===----------------
@HOOKS.register_module()
class TrainVisualizationHook(Hook):
    def __init__(self, interval=5, num_samples=3):
        self.interval = interval
        self.num_samples = num_samples

    def after_train_epoch(self, runner):
        if (runner.epoch + 1) % self.interval != 0:
            return

        # 安全获取数据加载器，避免迭代器异常
        train_dataloader = runner.train_dataloader
        data_iter = iter(train_dataloader)
        visualizer = runner.visualizer
        save_dir = os.path.join(runner.work_dir, 'visualizations')
        os.makedirs(save_dir, exist_ok=True)

        count = 0
        while count < self.num_samples:
            try:
                data_batch = next(data_iter)
            except StopIteration:
                break

            # 1. 严格提取模型所需输入（张量格式，移到GPU）
            inputs = data_batch['inputs']  # list[Tensor] -> 堆叠为batch张量
            if isinstance(inputs, list):
                inputs = torch.stack(inputs).to(runner.device, non_blocking=True)

            # 2. 模型推理（禁用梯度，确保格式正确）
            runner.model.eval()  # 切换评估模式
            with torch.no_grad():
                outputs = runner.model(inputs, mode='predict')
            runner.model.train()  # 切回训练模式

            # 3. 可视化（确保data_sample和output一一对应，避免索引错误）
            for idx, (data_sample, output) in enumerate(zip(data_batch['data_samples'], outputs)):
                if count >= self.num_samples:
                    break
                img_path = data_sample.get('img_path', '')
                if not img_path or not os.path.exists(img_path):
                    count += 1
                    continue

                # 加载原始图片，绘制标注和预测
                img = visualizer._load_image(img_path)
                # 真实标注图
                gt_img = visualizer.draw_datasample(
                    img.copy(), data_sample, draw_pred=False, show=False
                )
                # 预测结果图
                pred_img = visualizer.draw_datasample(
                    img.copy(), output, draw_gt=False, show=False
                )
                # 拼接保存
                combined_img = visualizer.hstack_imgs([gt_img, pred_img])
                save_path = os.path.join(save_dir, f'train_epoch_{runner.epoch+1}_sample_{count}.jpg')
                visualizer.imwrite(combined_img, save_path)
                count += 1

        print(f"📊 训练可视化已保存到：{save_dir}")

# ----------------=== 手动注册可视化器（避免未注册错误）===----------------
VISUALIZERS.register_module(name='DetLocalVisualizer', module=DetLocalVisualizer)

# ==================== 基础配置（路径均可用环境变量覆盖，不写死本机路径）====================
DATA_ROOT = os.environ.get('FS_COCO_ROOT', r'E:\dataset\fs_coco')
CLASSES = ('fire', 'smoke')
PRETRAINED_WEIGHTS = os.environ.get(
    'FS_PRETRAIN', 'weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
BATCH_SIZE = 8  # RTX 2060 6GB 适配
NUM_EPOCHS = 20
IMAGE_SIZE = (480, 320)  # 平衡速度和精度
work_dir = os.environ.get('FS_WORK_DIR', 'work_dirs/fire_smoke_detection')

# ==================== 模型配置（Faster R-CNN R50-FPN，COCO 预训练改 2 分类）====================
cfg = Config.fromfile(os.environ.get(
    'FS_MMDET_CONFIG',
    'mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'))

# 1. 数据集配置（适配COCO格式，Windows兼容）
cfg.data_root = DATA_ROOT
cfg.train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=DATA_ROOT,
        ann_file='annotations/train_annotations.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=CLASSES),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=IMAGE_SIZE, keep_ratio=True),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackDetInputs', meta_keys=['img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'])
        ]
    )
)

cfg.val_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=DATA_ROOT,
        ann_file='annotations/val_annotations.json',
        data_prefix=dict(img='val/'),
        metainfo=dict(classes=CLASSES),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=IMAGE_SIZE, keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs', meta_keys=['img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'])
        ]
    )
)

cfg.test_dataloader = cfg.val_dataloader

# 2. 模型适配（2分类+混合精度）
cfg.model.roi_head.bbox_head.num_classes = len(CLASSES)
cfg.model.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32
)

# 3. 训练参数优化（高效收敛）
cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=NUM_EPOCHS, val_interval=2)
# 替换为 AmpOptimWrapper，支持混合精度训练
cfg.optim_wrapper = dict(
    type='AmpOptimWrapper',  # 关键：使用混合精度优化器包装器
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale='dynamic'  # 动态损失缩放（混合精度必备）
)
cfg.param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', milestones=[10, 18], gamma=0.1, by_epoch=True)
]

# 4. 评估+保存配置（匹配指标键名）
cfg.val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(DATA_ROOT, 'annotations', 'val_annotations.json'),
    metric=['bbox'],
    outfile_prefix=os.path.join(work_dir, 'val_results')
)
cfg.test_evaluator = cfg.val_evaluator

cfg.default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/bbox_mAP_50',  # 与日志指标完全匹配
        rule='greater',
        interval=2,
        max_keep_ckpts=2,
        out_dir=work_dir
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# 5. 自定义钩子（仅保留必要功能）
@HOOKS.register_module()
class ConvergenceVisualizationHook(Hook):
    def __init__(self, save_interval=1):
        self.save_interval = save_interval  # 每多少个epoch保存一次图表
        self.train_losses = []  # 存储每个epoch的平均loss
        self.val_maps = []      # 存储每个epoch的mAP_50
        self.epochs = []        # 存储epoch序号

    def after_train_epoch(self, runner):
        """训练epoch结束后，收集训练loss"""
        # 从日志中获取当前epoch的平均total_loss（runner.log_buffer中存储了该epoch的统计值）
        avg_loss = runner.log_buffer.output['total_loss']
        self.train_losses.append(avg_loss)
        self.epochs.append(runner.epoch + 1)  # epoch从0开始，显示从1开始

    def after_val_epoch(self, runner, metrics):
        """验证epoch结束后，收集mAP并绘制图表"""
        current_epoch = runner.epoch + 1
        if current_epoch % self.save_interval != 0:
            return

        # 收集验证集mAP_50（metrics中存储了CocoMetric的评估结果）
        val_map_50 = metrics.get('coco/bbox_mAP_50', 0.0)
        self.val_maps.append(val_map_50)

        # 创建保存文件夹
        save_dir = os.path.join(runner.work_dir, 'convergence_plots')
        os.makedirs(save_dir, exist_ok=True)

        # 绘制双轴图表（loss和mAP在同一图中，不同y轴）
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

        # 绘制训练loss曲线（左y轴）
        ax1.plot(self.epochs, self.train_losses, color='#ff6b6b', linewidth=2.5, marker='o', label='Total Loss')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Training Loss', color='#ff6b6b', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#ff6b6b')
        ax1.grid(True, alpha=0.3)

        # 创建右y轴，绘制mAP曲线
        ax2 = ax1.twinx()
        ax2.plot(self.epochs[:len(self.val_maps)], self.val_maps, color='#4ecdc4', linewidth=2.5, marker='s', label='mAP@50')
        ax2.set_ylabel('Validation mAP@50', color='#4ecdc4', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#4ecdc4')
        ax2.set_ylim(0, 1.05)  # mAP范围0-1，留一点余量

        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

        # 添加标题和网格
        plt.title(f'Convergence Curve (Epoch {current_epoch})', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        # 保存图片（覆盖式保存，最新的图就是当前收敛状态）
        save_path = os.path.join(save_dir, 'convergence_curve.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"📈 收敛曲线已保存到：{save_path}")
# ==================== 工作目录+权重配置（保留原始路径）====================
cfg.work_dir = work_dir
cfg.load_from = PRETRAINED_WEIGHTS
cfg.resume = False

# ==================== 保存配置文件+启动训练 ====================
os.makedirs(work_dir, exist_ok=True)
config_save_path = os.path.join(work_dir, 'final_config.py')
cfg.dump(config_save_path)
print(f"📄 最终配置文件已保存到：{config_save_path}")

if __name__ == '__main__':
    # GPU环境验证
    print("="*60)
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | 显存: {torch.cuda.get_device_properties(0).total_memory/1024/1024:.0f}MB")
    else:
        print("⚠️  未检测到GPU，训练会很慢！")
    print("="*60)

    # 启动训练（无格式错误，直接运行）
    runner = Runner.from_cfg(cfg)
    runner.train()