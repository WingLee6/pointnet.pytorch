'''
### 说明文档

#### 代码作用

此代码实现了一个基于PointNet的3D点云分类模型的训练和评估过程。代码包含以下主要功能：

1. **参数设置**：使用 `argparse` 模块定义和解析命令行参数，参数包括批处理大小、工作线程数量、训练轮数、输出文件夹、模型路径、数据集路径、类别选择和是否使用特征变换。

2. **数据加载**：使用 `ShapeNetDataset` 类加载训练集和测试集数据，创建相应的数据加载器。

3. **模型初始化**：初始化 PointNet 分类器，并根据需要加载预训练模型。定义优化器（Adam）和学习率调度器。

4. **训练过程**：进行模型训练，包括损失计算和反向传播。每个训练批次后，打印训练损失和准确率，并每10个批次在测试集上进行一次评估。

5. **模型保存**：每个训练轮结束后，保存当前的模型参数。

6. **评估过程**：在测试集上计算每个类别的平均交并比 (mIOU)，用于评估模型在不同类别上的表现。

#### 输出结果

1. **终端输出**：
   - 随机种子
   - 使用的设备（CPU或GPU）
   - 数据集和测试集的大小
   - 类别数量
   - 训练损失和准确率
   - 测试损失和准确率
   - 每个类别的 mIOU

2. **模型文件**：
   - 训练过程中每个轮次结束时保存的模型参数文件，文件名格式为 `seg_model_<class_choice>_<epoch>.pth`，存储在指定的输出文件夹中。

#### 运行后的结果

1. **训练输出**：
   - 每个训练周期内，打印每个批次的训练损失和准确率。
   - 每10个批次，打印测试集上的损失和准确率。

2. **模型保存**：
   - 在指定的输出文件夹中保存模型参数文件，每个文件对应一个训练周期的模型状态。

3. **评估结果**：
   - 计算并打印最终的平均 mIOU，显示在终端中。mIOU 是各个类别的平均交并比，用于衡量模型在不同类别上的分割效果。

'''

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def main():
    # 创建参数解析器并定义所需的命令行参数
    parser = argparse.ArgumentParser()

    # 定义 batch size 参数
    # 默认 32
    parser.add_argument(
        '--batchSize', type=int, default=40, help='输入的 batch 大小') 

    # 定义数据加载 workers 的数量
    parser.add_argument(
        '--workers', type=int, help='数据加载 workers 的数量', default=4)

    # 定义训练 epoch 的数量
    # 默认 25
    parser.add_argument(
        '--nepoch', type=int, default=1, help='训练的 epoch 数量')

    # 定义输出文件夹
    parser.add_argument(
        '--outf', type=str, default='seg', help='输出文件夹')

    # 定义模型路径
    parser.add_argument(
        '--model', type=str, default='', help='模型路径')

    # 定义数据集路径（必需参数）
    parser.add_argument(
        '--dataset', type=str, required=True, help="数据集路径")

    # 定义类别选择，默认为 'Chair'
    parser.add_argument(
        '--class_choice', type=str, default='Chair', help="选择的类别")

    # 是否使用 feature transform
    parser.add_argument(
        '--feature_transform', action='store_true', help="是否使用 feature transform")

    # 解析命令行参数
    opt = parser.parse_args()
    print(opt)

    # 设置随机种子以保证实验可复现性
    opt.manualSeed = random.randint(1, 10000)
    print("随机种子: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # 检测是否有可用的 GPU，如果没有则使用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用设备: ", device)

    # 加载训练数据集
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=False,
        class_choice=[opt.class_choice])

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    # 加载测试数据集
    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=False,
        class_choice=[opt.class_choice],
        split='test',
        data_augmentation=False)

    # 创建测试数据加载器
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    # 输出数据集和测试集的大小
    print(len(dataset), len(test_dataset))

    # 获取分割类别的数量
    num_classes = dataset.num_seg_classes
    print('类别数量', num_classes)

    # 尝试创建输出文件夹
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    # 定义蓝色字体输出（用于终端显示）
    blue = lambda x: '\033[94m' + x + '\033[0m'

    # 初始化分类器（分割任务）
    classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

    # 如果指定了模型路径，加载预训练模型的参数
    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    # 定义优化器（Adam）和学习率调度器
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 将模型移动到设备（GPU/CPU）
    classifier = classifier.to(device)

    # 计算每个 epoch 的批次数量
    num_batch = len(dataset) / opt.batchSize

    # 训练循环
    for epoch in range(opt.nepoch):
        scheduler.step()  # 更新学习率
        for i, data in enumerate(dataloader, 0):
            points, target = data  # 获取输入数据和目标标签
            points = points.transpose(2, 1)  # 转置点云数据的维度以符合网络输入格式
            points, target = points.to(device), target.to(device)  # 将数据移动到设备
            optimizer.zero_grad()  # 梯度清零
            classifier = classifier.train()  # 设置模型为训练模式
            pred, trans, trans_feat = classifier(points)  # 前向传播
            pred = pred.view(-1, num_classes)  # 调整预测结果的形状
            target = target.view(-1, 1)[:, 0] - 1  # 调整目标标签的形状并减 1 以匹配类别索引
            loss = F.nll_loss(pred, target)  # 计算损失
            if opt.feature_transform:  # 如果使用 feature transform，增加正则化损失
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            pred_choice = pred.data.max(1)[1]  # 选择最大概率的类别作为预测结果
            correct = pred_choice.eq(target.data).cpu().sum()  # 计算正确预测的数量
            print('[%d: %d/%d] 训练损失: %f 准确率: %f' % (
                epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize * 2500)))

            # 每 10 个批次进行一次测试集上的评估
            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.to(device), target.to(device)
                classifier = classifier.eval()  # 设置模型为评估模式
                pred, _, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1
                loss = F.nll_loss(pred, target)  # 计算测试集上的损失
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s 损失: %f 准确率: %f' % (
                    epoch, i, num_batch, blue('测试'), loss.item(), correct.item() / float(opt.batchSize * 2500)))

        # 每个 epoch 结束后保存模型
        torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

    # 计算分割任务的 mIOU
    shape_ious = []
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        classifier = classifier.eval()  # 设置模型为评估模式
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(2)[1]

        # 将预测结果和真实标签转换为 numpy 数组
        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1

        # 逐形状计算 IOU
        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1  # 如果预测和真实标签均为空，则 IOU 设为 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

    # 输出 mIOU 结果
    print("类 {} 的 mIOU: {}".format(opt.class_choice, np.mean(shape_ious)))


if __name__ == '__main__':
    main()
