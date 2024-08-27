'''
README:

运行该代码后，会生成以下内容：
1. 模型权重文件: 
    - 每个 epoch 结束时，当前模型的权重会被保存到指定的输出文件夹 (`opt.outf`) 中，文件名格式为 `cls_model_{epoch}.pth`。
        例如，若指定的输出文件夹是 `cls`，在第 10 个 epoch 后保存的模型文件将是 `cls/cls_model_10.pth`。

2. 训练日志:
    - 在每个训练批次（`batch`）后，代码会打印当前的训练损失 (`loss`) 和准确率 (`accuracy`)。
    - 每 10 个批次，会在测试集上进行一次评估，并打印测试损失和准确率。
    - 这些日志信息会输出到终端中，但不会自动保存为文件。如果需要保存，可以使用文件重定向或增加日志记录功能。

3. 最终的测试准确率:
    - 在整个训练完成后，代码会在测试集上评估模型的最终准确率，并将结果打印在终端中。


运行示例:
```
python train_classification.py --dataset /Users/lee/Git\ Projects/pointnet.pytorch/datasets/shapenetcore_partanno_segmentation_benchmark_v0 --nepoch=1 --dataset_type shapenet
```
'''



from __future__ import print_function  # 兼容 Python 2 和 Python 3 的 print 函数
import argparse  # 用于解析命令行参数
import os  # 提供操作系统相关的功能
import random  # 提供生成随机数的功能
import torch  # PyTorch 主库，提供张量计算和深度学习功能
import torch.nn.parallel  # 用于多 GPU 并行计算的工具
import torch.optim as optim  # 包含常用的优化器，如 Adam 和 SGD
import torch.utils.data  # 数据加载工具，支持批量加载和多线程加载
from pointnet.dataset import ShapeNetDataset, ModelNetDataset  # 导入自定义的 ShapeNet 和 ModelNet 数据集类
from pointnet.model import PointNetCls, feature_transform_regularizer  # 导入 PointNet 分类模型和特征变换正则化器
import torch.nn.functional as F  # 提供许多常用的函数式 API，如激活函数、损失函数等
from tqdm import tqdm  # 用于显示进度条

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=40, help='输入批次大小')   # 默认32
parser.add_argument('--num_points', type=int, default=2500, help='输入点云大小')
parser.add_argument('--workers', type=int, help='数据加载的工作线程数量', default=4)
parser.add_argument('--nepoch', type=int, default=2, help='训练的总轮数') # 默认250
parser.add_argument('--outf', type=str, default='cls', help='输出文件夹')
parser.add_argument('--model', type=str, default='', help='预训练模型路径')
parser.add_argument('--dataset', type=str, required=True, help="数据集路径")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="数据集类型 shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="使用特征变换")
opt = parser.parse_args()

# 打印解析的命令行参数
print(opt)

# 定义一个打印蓝色文本的函数
blue = lambda x: '\033[94m' + x + '\033[0m'

# 设置随机种子以确保实验的可重复性
opt.manualSeed = random.randint(1, 10000)  # 生成一个随机种子
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)  # 为 Python 自带的 random 模块设置种子
torch.manual_seed(opt.manualSeed)  # 为 PyTorch 设置种子

# 根据数据集类型加载数据集
if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points
    )

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False
    )

elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval'
    )

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False
    )

else:
    exit('错误的数据集类型')

# 创建数据加载器，用于批量加载训练数据
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers)
)

# 创建数据加载器，用于批量加载测试数据
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers)
)

# 打印训练集和测试集的大小
print(len(dataset), len(test_dataset))

# 获取分类的类别数
num_classes = len(dataset.classes)
print('类别数', num_classes)

# 尝试创建输出文件夹，如果文件夹已存在则忽略
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# 初始化 PointNet 分类模型
classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

# 如果指定了预训练模型的路径，则加载模型参数
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

# 自动检测 GPU 是否可用，如果可用则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 将模型移动到选定的设备
classifier = classifier.to(device)

# 定义 Adam 优化器
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))

# 定义学习率调度器，每 20 个 epoch 乘以 0.5
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 计算每个 epoch 的批次数
num_batch = len(dataset) / opt.batchSize

# 主训练过程
if __name__ == '__main__':

    for epoch in range(opt.nepoch):  # 遍历每个 epoch
        scheduler.step()  # 更新学习率

        for i, data in enumerate(dataloader, 0):  # 遍历每个批次
            points, target = data  # 从数据加载器中获取点云和标签
            target = target[:, 0]  # 提取目标标签
            points = points.transpose(2, 1)  # 转置点云数据以匹配模型输入
            points, target = points.to(device), target.to(device)  # 将数据移动到设备上

            optimizer.zero_grad()  # 清除梯度
            classifier = classifier.train()  # 切换模型到训练模式

            pred, trans, trans_feat = classifier(points)  # 前向传播计算输出
            loss = F.nll_loss(pred, target)  # 计算负对数似然损失

            if opt.feature_transform:  # 如果使用特征变换，则加上正则化损失
                loss += feature_transform_regularizer(trans_feat) * 0.001

            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            pred_choice = pred.data.max(1)[1]  # 预测的类别
            correct = pred_choice.eq(target.data).cpu().sum()  # 计算正确的预测数量

            print('[%d/%d: %d/%d] 损失: %f 准确率: %f' % (
                epoch, opt.nepoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize))
            )

            if i % 10 == 0:  # 每 10 个批次进行一次测试
                j, data = next(enumerate(testdataloader, 0))  # 获取下一个测试批次
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.to(device), target.to(device)  # 将数据移动到设备上

                classifier = classifier.eval()  # 切换模型到评估模式
                pred, _, _ = classifier(points)  # 前向传播计算输出

                loss = F.nll_loss(pred, target)  # 计算测试损失
                pred_choice = pred.data.max(1)[1]  # 预测的类别
                correct = pred_choice.eq(target.data).cpu().sum()  # 计算正确的预测数量

                print('[%d/%d: %d/%d] %s 损失: %f 准确率: %f' % (
                    epoch, opt.nepoch, i, num_batch, blue('测试'), loss.item(), correct.item() / float(opt.batchSize))
                )

        # 在每个 epoch 结束后保存模型
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    # 评估模型在测试集上的最终性能
    total_correct = 0
    total_testset = 0

    for i, data in tqdm(enumerate(testdataloader, 0)):  # 遍历每个测试批次
        points, target = data  # 获取点云和标签
        target = target[:, 0]  # 提取目标标签
        points = points.transpose(2, 1)  # 转置点云数据
        points, target = points.to(device), target.to(device)  # 将数据移动到设备上

        classifier = classifier.eval()  # 切换模型到评估模式
        pred, _, _ = classifier(points)  # 前向传播计算输出

        pred_choice = pred.data.max(1)[1]  # 预测的类别
        correct = pred_choice.eq(target.data).cpu().sum()  # 计算正确的预测数量

        total_correct += correct.item()  # 累加正确的预测数量
        total_testset += points.size()[0]  # 累加测试样本数量

    # 打印最终的测试准确率
    print("最终测试准确率: %f" % (total_correct / float(total_testset)))
