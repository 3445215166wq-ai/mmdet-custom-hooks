# mmdet 3.x 自定义 Hook：训练过程可视化

在 [mmengine Runner](https://mmengine.readthedocs.io/) 的生命周期上挂两个自定义 Hook，
解决 mmdet 默认不给的两样东西：**「真值框 vs 预测框」对比图** 和 **「loss 与验证 mAP 画在同一张收敛曲线」**。

落地场景是一个烟火检测（fire / smoke）训练任务。**mAP@50 只有 0.211** —— 但 [§7.5](#75-map-低--模型差抽查最差图之后我把结论改了) 把预测框按阈值拆开、再把最差的 6 张图逐张叠加真值与预测看过之后，**结论被推翻了：模型其实定位得很准，是这个公开数据集的标注严重不全**，`0.211` 不能当作模型能力的数字。仓库的价值在前两节（Hook 的写法与踩过的坑）和 §7.5（**一次把自己结论推翻的排查过程**）。

---

## 1. 为什么得自己写 Hook

mmdet 3.x 用 mmengine 的 Runner 组织训练循环。默认的 `LoggerHook` / `VisualizationHook` 只把指标写日志，**不会把某一帧的 GT 框和预测框并排存成图**；而 loss 在训练阶段、mAP 在验证阶段产生，**两者生命周期不同，日志里永远不在同一行**——想画双轴收敛曲线，只能在 Hook 里各收一半。

mmengine 的 Hook 有一组固定回调点，本次用到四个：

| 回调点 | 触发时机 | 这里用来做什么 |
|---|---|---|
| `before_run` / `after_run` | 整个训练前后 | （未使用） |
| **`after_train_epoch`** | 每个训练 epoch 结束 | Hook 1 存对比图；Hook 2 收集 `total_loss` |
| **`after_val_epoch`** | 每个验证 epoch 结束 | Hook 2 收集 `coco/bbox_mAP_50` 并绘图 |

---

## 2. Hook 1：`TrainVisualizationHook` —— 每 N 轮存一组「真值 vs 预测」

```python
@HOOKS.register_module()
class TrainVisualizationHook(Hook):
    def __init__(self, interval=5, num_samples=3):
        self.interval = interval
        self.num_samples = num_samples

    def after_train_epoch(self, runner):
        if (runner.epoch + 1) % self.interval != 0:
            return
        data_iter = iter(runner.train_dataloader)
        visualizer = runner.visualizer
        save_dir = os.path.join(runner.work_dir, 'visualizations')
        os.makedirs(save_dir, exist_ok=True)
        # ... 取 batch → 推理 → 拼图 → 存盘（完整见 fire_smoke_detection.py）
```

四个不能省的细节：

1. **必须 `runner.model.eval()` + `torch.no_grad()`**，结束后再 `train()` 切回。否则在训练模式下做 forward 会带着 dropout/BN 的随机性，而且会往计算图里挂东西。
2. **`inputs` 在 batch 里是 `list[Tensor]`**（每张图尺寸不同），要用 `torch.stack` 堆成 batch 张量再送 GPU——直接 `data_batch['inputs'].to(device)` 会报 `list has no attribute to`。
3. **`data_sample` 和 `output` 必须一一对应**，用 `zip()` 同时遍历两个列表；两者数量不等时（模型丢样本）要能安全退出。
4. **拼图用 `visualizer.hstack_imgs([gt_img, pred_img])`**，左边真值右边预测——这是 DetLocalVisualizer 自带的方法，比自己写 `np.hstack` 稳妥。

输出：`{work_dir}/visualizations/train_epoch_{N}_sample_{i}.jpg`

---

## 3. Hook 2：`ConvergenceVisualizationHook` —— loss 与 mAP 同图双轴

```python
@HOOKS.register_module()
class ConvergenceVisualizationHook(Hook):
    def after_train_epoch(self, runner):
        # loss 只在这里拿得到
        self.train_losses.append(runner.log_buffer.output['total_loss'])
        self.epochs.append(runner.epoch + 1)

    def after_val_epoch(self, runner, metrics):
        # mAP 只在这里拿得到
        self.val_maps.append(metrics.get('coco/bbox_mAP_50', 0.0))
        # ... matplotlib 双 y 轴绘图，覆盖式保存 convergence_curve.png
```

关键点：

- **`after_train_epoch` 里没有 `metrics`，`after_val_epoch` 里没有 `total_loss`**——这是把 Hook 拆成两个回调方法的根本原因，不是风格问题。`total_loss` 要从 `runner.log_buffer.output` 里取（LoggerHook 已经把本轮统计写进去了）。
- **`metrics.get('coco/bbox_mAP_50', 0.0)`** 的键名必须和 config 里 `CocoMetric(metric=['bbox'])` 产出的键完全一致，写错不会报错，只会静默拿到 0。
- 验证不是每轮都跑（本项目 `val_interval=2`），所以 `self.epochs` 与 `self.val_maps` 长度不等，绘图时要 `self.epochs[:len(self.val_maps)]` 截断对齐。
- 覆盖式保存同一种文件名而不是逐轮追加，好处是随时打开就是当前收敛状态，不会堆几十张图。

输出：`{work_dir}/convergence_plots/convergence_curve.png`

---

## 4. 两个必须处理的 registry 坑

mmdet 3.x 把模块拆到多个包，直接跑 `Runner.from_cfg()` 常撞这两条：

```python
# ① 数据集/模型类没注册 → Config 解析时报 KeyError: 'XxxDataset' is not in the registry
from mmdet.utils import register_all_modules
register_all_modules()

# ② 可视化器不在默认可视化器 registry 里 → 自写 Hook 里 runner.visualizer 拿不到 DetLocalVisualizer
from mmengine.registry import VISUALIZERS
from mmdet.visualization import DetLocalVisualizer
VISUALIZERS.register_module(name='DetLocalVisualizer', module=DetLocalVisualizer)
```

第 ② 条是这次真正卡住的地方：`DetLocalVisualizer` 定义在 `mmdet.visualization`，而默认 visualizer registry 指向 mmengine 那一侧；`register_all_modules()` 解决的是 dataset/model 的注册，**不解决 visualizer**，必须手动补一句。

---

## 5. 数据集

**公开的烟火检测数据集**，两类 `fire` / `smoke`，已整理成 mmdet 需要的 COCO 格式（`annotations/{train,val}_annotations.json`）。图片名为 `fire_XXXXXX.jpg`，尺寸不统一（277×372 ~ 913×599，`keep_ratio=True` 所以每张真实输入尺寸都不同）。数据不随仓库提供。

| 划分 | 图片 | 标注框 | 无标注图 | 平均框/图 | fire 框 | smoke 框 |
|---|---:|---:|---:|---:|---:|---:|
| train | 5,552 | 20,845 | 206 | 3.75 | 10,812 | 10,033 |
| val | 1,388 | 5,166 | 59 | 3.72 | 2,683 | 2,483 |
| 合计 | **6,940** | **26,011** | 265 | 3.73 | 13,495 | 12,516 |

两类框数接近 1:1，所以下面的低指标**不是类别不平衡导致的**。

---

## 6. 训练配置（实际跑的那一份）

| 项 | 值 |
|---|---|
| 模型 | Faster R-CNN R50-FPN（COCO 预训练），`bbox_head.num_classes=2` |
| 输入 | `(480, 320)`，`keep_ratio=True` |
| batch | 8（RTX 2060 6 GB） |
| 混合精度 | `AmpOptimWrapper` + `loss_scale='dynamic'`，`clip_grad(max_norm=35)` |
| 优化器 | SGD `lr=1e-3`, `momentum=0.9`, `weight_decay=1e-4` |
| 调度 | `LinearLR` 热身 500 iter → `MultiStepLR` milestones `[10, 18]`，`gamma=0.1` |
| 循环 | `EpochBasedTrainLoop`，`max_epochs=20`，`val_interval=2` |
| 保存 | `CheckpointHook(save_best='coco/bbox_mAP_50', rule='greater', interval=2, max_keep_ckpts=2)` |

---

## 7. 结果（真实，不美化）

逐轮验证（`val_interval=2`，测试集 694 iter，batch 2）：

| epoch | mAP | mAP@50 | mAP@75 | mAP_s | mAP_m | mAP_l |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.057  | 0.157  | 0.030 | 0.023 | 0.047 | 0.098 |
| 4 | 0.070  | 0.188  | 0.041 | 0.025 | 0.055 | 0.126 |
| 6 | 0.074  | 0.197  | 0.042 | 0.035 | 0.058 | 0.137 |
| 8 | 0.078  | 0.202  | 0.047 | 0.028 | 0.061 | 0.141 |
| 10 | 0.078  | 0.204  | 0.045 | 0.037 | 0.062 | 0.142 |
| **12** | **0.082** | **0.211** | 0.051 | 0.032 | 0.064 | 0.148 |
| 14 | 0.081  | 0.208  | 0.050 | 0.033 | 0.065 | 0.147 |
| 16 | 0.080  | 0.209  | 0.048 | 0.033 | 0.067 | 0.147 |
| 18 | 0.080  | 0.210  | 0.049 | 0.034 | 0.065 | 0.148 |
| 20 | 0.080  | 0.208  | 0.050 | 0.033 | 0.065 | 0.148 |

取最佳权重（epoch 12，`coco/bbox_mAP_50=0.211`）在测试集上复核，指标同值：**mAP 0.082 / mAP@50 0.211**。
原始日志：[`logs/train_full_20251108_083125.log`](logs/train_full_20251108_083125.log)（20 轮完整训练，约 2.5 小时）、
[`logs/train_20251108_130024.log`](logs/train_20251108_130024.log)（最佳权重单独测试）。日志里的本机用户名与主机名已替换为占位符。

### 结论：**不是训练不够久，是小目标**

三点观察，都来自上表：

1. **第 12 轮就饱和了。** 从 0.0781（ep10）爬到 0.0820（ep12），之后 8 轮全部在 0.080–0.081 之间震荡，期间学习率还在 ep18 又降了一次——**没有任何提升**。所以"加轮数"不是解法，跑满 20 轮里后 8 轮其实是白跑。
2. **`mAP_s`（小目标）= 0.032，`mAP_l`（大目标）= 0.148，差 4.6 倍。** 输入被压到 480×320 且 `keep_ratio=True`，远处的烟柱缩到几十像素——特征在 backbone 里确实会丢。**但这条要打折看**：小目标恰恰是最容易被漏标的（见 §7.5），`mAP_s` 低里既有模型的原因也有标注的原因，不能全算在分辨率头上。
3. **两类框数接近 1:1**，`mAP@75` 却只有 0.051，说明框的位置回归也很松。火焰与烟雾本身边界模糊、同一片烟不同人标的范围能差一倍，这更像是**标注一致性**问题而不是模型容量问题。

所以我当时的判断是：下一步该做的是 ① 输入提到 640 或按原图分辨率滑窗切片推理 ② 抽 50 张图人工复核标注一致性 ③ 换 anchor-free 的多尺度模型（RT-DETR / YOLOv8）对比，**而不是继续调学习率或加轮数**。

---

## 7.5 mAP 低 ≠ 模型差——抽查最差图之后我把结论改了

mAP 是把所有置信度上的表现平均掉的一个数，**会掩盖"某个阈值区间内其实可用"这件事**。上一版权重留下了完整的 44,435 个预测框（`test_results.bbox.json`），于是我按 IoU≥0.5、同类别的贪心匹配，重新按阈值拆了一遍：

| 置信度阈值 | TP | FP | FN | 精确率 | 召回率 | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 3,399 | 40,149 | 1,767 | 0.078 | 0.658 | 0.140 |
| 0.10 | 3,036 | 23,100 | 2,130 | 0.116 | 0.588 | 0.194 |
| 0.20 | 2,524 | 11,627 | 2,642 | 0.178 | 0.489 | 0.261 |
| 0.30 | 2,114 | 6,888 | 3,052 | 0.235 | 0.409 | 0.298 |
| **0.40** | 1,752 | 4,333 | 3,414 | **0.288** | **0.339** | **0.311**（F1 最优） |
| 0.50 | 1,331 | 2,616 | 3,835 | 0.337 | 0.258 | 0.292 |
| 0.70 | 590 | 630 | 4,576 | 0.484 | 0.114 | 0.185 |
| 0.90 | 128 | 32 | 5,038 | **0.800** | 0.025 | 0.048 |

**图片级**（有真值的 1,329 张图里，至少被正确框中一个的比例）：

| 阈值 | 至少命中 1 个框 | 占比 |
|---:|---:|---:|
| 0.10 | 1,210 | **91.0%** |
| 0.20 | 1,167 | 87.8% |
| 0.30 | 1,095 | 82.4% |
| 0.50 | 895 | 67.3% |

**怎么读这两张表——以及我一开始读错的地方**

第一版 README 里我把这张表读成"框海战术：命中 1 个真框要附带约 12 个假框，所以不能当可用"。**这个结论是错的**，错在把"与标注不重叠"当成了"误检"。

我把背景误检最多的 6 张图（`smoke_000029` / `fire_detection_cfd13057…` / `smoke_01159` / `fire_00583` / `smoke_001603` / `fire_000963`）把真值画绿、预测画红叠在一起逐张看过，结果一致：

- **预测框（红，score ≥ 0.5）全部落在真实的火焰与烟雾上**——楼里一扇扇窗户往外冒的火苗、顶部的黑色烟柱、墙后的烟团，都框到了，分数 0.53~0.82；
- **没有一处框到天空、墙面、消防梯、草地或人群**；
- 而**绿框（真值）只标了整幅画面的几个大区域**：`smoke_000029` 拍到 16 个预测框，真值只有 4 个；`fire_detection_cfd13057…` 拍出 9 个预测框，真值只有 **1 个**；`fire_00583` 真值 11 个、预测 10 个，两者基本对得上。

⇒ **"背景误检"这个桶里混进了大量"检测正确但没被标注"的框**。这不是模型在乱框，是数据集漏标。

**数据集是多个公开来源拼起来的**，文件名就能看出来：`fire_000003.jpg`（连号）、`smoke_000029.jpg`、`smoke_fire_378.jpg`、`fire_detection_<uuid>.jpg`（Roboflow 导出的命名）。多源拼接意味着**标注口径不一致**——有的图只框主火区、有的图逐簇火焰标注，这也是 `mAP@75` 只有 0.051 的原因：框归一化的标准本身就不统一。

**所以真正站得住的结论是这三条：**

1. **模型能把画面里的火焰与烟雾定位出来。** 视觉抽查最差的 6 张图，零误检、火点全部覆盖。
2. **但拿不出可信的准确率 / 召回率。** 不是模型没能力，是**标签不可信**——`mAP 0.211` 是"对这份标注的成绩"，不能当作模型能力的数字；同样也不能反过来引用"框级精确率 0.078"说它乱框。
3. **这也解释了为什么第 12 轮就饱和**：模型在拟合一份自相矛盾的标签，再训练也压不动损失。

**下一步该做的是修标注**（补标漏掉的火焰/烟雾 + 统一多源标注口径），而不是调参或换模型。这张阈值表的用处是"有了干净标注之后，用什么阈值配什么用途"，前提是先有可信的 GT。

---

## 8. 移植到自己的检测任务

```python
# config 里挂上即可，不需要改 mmdet 源码
custom_hooks = [
    dict(type='TrainVisualizationHook', interval=5, num_samples=3, priority='NORMAL'),
    dict(type='ConvergenceVisualizationHook', save_interval=1, priority='NORMAL'),
]
```

前提：两个 Hook 所在的模块要**先被 import**（继承 `Hook` 并用 `@HOOKS.register_module()` 注册）。
最省事的做法就是像本项目这样，把 Hook 和 config 写在同一个脚本里，脚本顶部 import 一次。

---

## 9. 附带的日志后处理脚本 `visual.py`

mmdet 的日志里，`Epoch(train)` 的 loss 和 `Epoch(val)` 的 mAP **分行输出且不保证相邻**，直接按行号配对会把指标错配到别的轮次。脚本的做法：

- 用两条正则分别抓 `Epoch(train) [N]` 的 loss 与 `Epoch(val) [N]` 的 mAP / mAP@50；
- **要求 `val_epoch == 当前 train_epoch` 才配对**，配对成功就把两个计数器清空——这是防轮次错位的关键；
- 日志是 **GBK** 编码，`open(..., encoding='gbk')` 显式指定，否则中文全乱码。

```bash
python visual.py --log logs/train_full_20251108_083125.log --out metrics_curve.png
```

> 这个脚本是写得比较早的版本，用 pandas 但最后只用了 matplotlib。现在看直接读日志建 `dict` 更清楚——留着是为了记录当时踩过的编码和错位这两个坑。

---

## 10. 已知问题（不回避）

1. **`TrainVisualizationHook` 在 `after_train_epoch` 里对 `runner.train_dataloader` 重新 `iter()`**，而训练循环自己还握着一个迭代器；config 里 `persistent_workers=True` 时两者会抢同一批 worker，极端情况下打乱该 epoch 的取数顺序。更稳的写法是在 Hook 里另建一个只取 N 张的 Dataset，或缓存 loop 里第一批 batch 的引用。
2. 只做了可视化，**没有把预测结果落盘成 COCO json** 做离线错误分析。
3. 没有做 `score_thr` / NMS 阈值的调参对比，图表里的框是默认阈值下画的。
4. 数据集是**公开的烟火检测数据集**（6,940 张，已整理成 COCO 格式），**没有做标注一致性交叉校验**——这大概率是指标上不去的隐藏原因。
5. `mAP@50 = 0.211` 这个绝对值不能与公开数据集上的结果横向比较：这是自建数据集、20 轮、480×320 输入下的数字。

---

## 11. 环境与复现

| 项 | 版本 |
|---|---|
| GPU | NVIDIA RTX 2060（6 GB） |
| PyTorch | 1.12.1+cu113 |
| TorchVision | 0.13.1+cu113 |
| MMEngine | 0.10.7 |
| mmdet | 3.x（`FasterRCNN` + `CocoDataset` + `DetLocalVisualizer`） |

```bash
# 1. 准备 mmdet 3.x 环境（按官方文档装 mmcv）
# 2. 数据集放成 COCO 结构：{root}/annotations/{train,val}_annotations.json + {root}/train|val/*.jpg
export FS_COCO_ROOT=/path/to/fs_coco
export FS_PRETRAIN=/path/to/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
export FS_MMDET_CONFIG=/path/to/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
python fire_smoke_detection.py
```

```
.
├── fire_smoke_detection.py    # 两个自定义 Hook + 完整训练配置 + Runner 启动
├── visual.py                  # 从日志正则提取 loss/mAP 并画双轴曲线
├── analyze_thresholds.py      # §7.5 那张按阈值拆的表，就是这个脚本算的
├── inspect_predictions.py     # 把真值(绿)与预测(红)叠在图上，人工核对误检——§7.5 的转折点靠它
├── results/
│   └── val_predictions.coco.json   # 6.5 MB，验证集全部 44,435 个预测框（COCO 结果格式）
└── logs/                      # 真实训练与测试日志（用户名已脱敏）
```

数据集的 6,940 张图片、预训练权重与训练出的 checkpoint **不随仓库提供**（数据集里的图片来源不一，
可能包含媒体照片，不适合再分发）；但预测结果已随仓库附带，§7.5 的两步分析可以完整复现：

```bash
# 第一步：按阈值拆开看统计
python analyze_thresholds.py \
    --pred results/val_predictions.coco.json \
    --gt /path/to/fs_coco/annotations/val_annotations.json

# 第二步：把最可疑的图叠出来人眼看（关键的一步）
python inspect_predictions.py \
    --gt /path/to/fs_coco/annotations/val_annotations.json \
    --img-dir /path/to/fs_coco/val \
    --out-dir viz --thr 0.5 \
    smoke_000029.jpg fire_000963.jpg
```
