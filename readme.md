# TSCAN

掌纹跨设备 / 跨场景识别实验仓库。当前代码以两阶段流程为主：

1. 源域教师网络初始化  
2. 基于教师网络的 TSCAN 跨域自适应训练

仓库里同时保留了对比实验脚本，包括：

- `TSCAN`
- `MMD-ResNet`
- `DANN`

以及用于论文绘图的 ROC、t-SNE 和验证曲线导出脚本。

## 数据集

默认使用本地 XJTU-UP ROI 数据，目录为：

```bash
/home/repository/PalmDatas/XJTUUP
```

当前代码实际使用的子库名称为：

- `HUAWEI/Nature`
- `HUAWEI/Flash`
- `iPhone/Nature`
- `iPhone/Flash`
- `MI8/Flash`
- `LG/Nature`
- `LG/Flash`
- `Samsung/Nature`
- `Samsung/Flash`

每个域目录下应包含：

```text
cent_train/
cent_test/
```

其中每个身份对应一个子目录，图像位于身份目录下。

## 主要脚本

### 1. 教师网络初始化

源域有监督训练脚本：

```bash
python train_one.py --config configs/train_configs/train_one.yaml
```

典型输出目录：

```text
results/train_one/final_teacher
```

教师网络跨域评估汇总：

```text
results/train_one/final_teacher/teacher_eval_summary.csv
```

### 2. TSCAN 主训练

四个协议的配置在：

```text
configs/new_train/
```

统一运行：

```bash
bash configs/new_train/run.sh
```

当前主实验结果目录：

```text
results/tscan_new_train/
```

### 3. 消融实验

协议 I 的模块消融配置在：

```text
configs/melt_configs/
```

统一运行：

```bash
bash configs/melt_configs/run.sh
```

当前消融结果目录包括：

```text
results/tscan_melt/
results/tscan_melt_v0/
results/tscan_melt_v1/
```

### 4. 置信度阈值实验

协议 I 的阈值实验配置在：

```text
configs/conf_configs/
```

协议 II / III / IV 的阈值实验配置在：

```text
configs/conf234_configs/
```

### 5. 对比实验

MMD 与 DANN 的最简对比训练配置在：

```text
configs/contrast_configs/
```

运行 MMD：

```bash
bash configs/contrast_configs/run_mmd.sh
```

运行 DANN：

```bash
bash configs/contrast_configs/run_dann.sh
```

统一运行：

```bash
bash configs/contrast_configs/run_all.sh
```

结果目录：

```text
results/contrast/mmd/
results/contrast/dann/
```

## ROC 绘图

四协议对比 ROC 使用：

```bash
bash configs/contrast_configs/run_roc.sh
```

当前脚本会生成：

- 四协议 2×2 ROC 总图
- 每条曲线的原始 `FAR/TAR` 数据 CSV

输出目录：

```text
results/contrast/plots/
```

其中：

- 总图：`results/contrast/plots/compare_roc_grid.png`
- 曲线数据：`results/contrast/plots/data/*.csv`

## TSCAN 训练逻辑

当前 `TSCAN` 训练脚本位于：

```text
DA/train_tscan.py
```

核心实现位于：

```text
DA/tscan.py
```

主要包含：

- `ResNet_double` 主干与 ArcFace 度量头
- 目标域伪标签刷新
- 教师网络 EMA 更新
- 域对抗分支
- t-SNE 可视化
- ROC / DET / 分数分布导出

## 评估口径

统一使用：

- `Acc`
- `EER`
- `TAR@FAR=0.1`
- `TAR@FAR=0.01`
- `TAR@FAR=0.001`
- `mAP`

底层评估函数位于：

```text
val_base.py
```

## 当前常用结果目录

```text
results/train_one/final_teacher
results/tscan_new_train
results/contrast
results/tscan_melt
results/tscan_conf
results/contrast/plots
```

## 说明

- 训练和绘图脚本默认依赖本地绝对路径数据集。
- 若更换设备目录名称，需同步修改配置文件中的 `source_path` / `target_path`。
- 对比实验与主实验当前采用相同 backbone 与相同评估口径，便于论文表格统一统计。
