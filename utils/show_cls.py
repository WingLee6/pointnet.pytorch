from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetCls
import torch.nn.functional as F

def main():
    # 创建参数解析器并定义所需的命令行参数
    parser = argparse.ArgumentParser()

    # 定义模型路径参数
    parser.add_argument('--model', type=str, default='', help='保存模型的路径')
    
    # 定义输入点云的数量参数
    parser.add_argument('--num_points', type=int, default=2500, help='输入点云的数量')

    # 解析命令行参数
    opt = parser.parse_args()
    print(opt)

    # 加载测试数据集
    test_dataset = ShapeNetDataset(
        root='shapenetcore_partanno_segmentation_benchmark_v0',
        split='test',
        classification=True,
        npoints=opt.num_points,
        data_augmentation=False
    )

    # 创建测试数据加载器
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True
    )

    # 初始化分类器模型
    classifier = PointNetCls(k=len(test_dataset.classes))
    
    # 检测是否有可用的 GPU，如果没有则使用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用设备: ", device)

    # 将模型移动到设备（GPU/CPU）
    classifier = classifier.to(device)
    
    # 加载模型的预训练参数
    classifier.load_state_dict(torch.load(opt.model))
    
    # 设置模型为评估模式
    classifier.eval()

    # 在测试数据上进行评估
    for i, data in enumerate(testdataloader, 0):
        points, target = data
        
        # 将数据转换为变量，并移动到设备
        points, target = Variable(points).to(device), Variable(target[:, 0]).to(device)
        
        # 转置点云数据以符合网络输入格式
        points = points.transpose(2, 1)
        
        # 前向传播
        pred, _, _ = classifier(points)
        
        # 计算损失
        loss = F.nll_loss(pred, target)
        
        # 获取预测结果
        pred_choice = pred.data.max(1)[1]
        
        # 计算正确预测的数量
        correct = pred_choice.eq(target.data).cpu().sum()
        
        # 打印损失和准确率
        print('i:%d  loss: %f accuracy: %f' % (i, loss.data.item(), correct / float(32)))
        
if __name__ == '__main__':
    main()
