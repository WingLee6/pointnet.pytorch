from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls
import matplotlib.pyplot as plt


import yaml
import os

# 从配置文件加载项目相关的配置信息
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 获取项目的根目录路径
project_directory = config['project_directory']

# 根据配置文件中的相对路径构建数据集路径、训练结果路径和检查点路径
dataset_path = os.path.join(project_directory, config['paths']['dataset_path'])
classifier_model_path = os.path.join(project_directory, config['paths']['classifier_model_path'])
segmentation_model_path = os.path.join(project_directory, config['paths']['segmentation_model_path'])
print("数据集路径: ", dataset_path)
print("分类器模型路径: ", classifier_model_path)
print("分割模型路径: ", segmentation_model_path)

# 分割的类别
class_choice = 'Mug'

def main(): 
    

    # 加载 ShapeNet 数据集
    # 命令行参数解析
    parser = argparse.ArgumentParser()

    # 模型路径
    parser.add_argument('--model', type=str, default=segmentation_model_path, help='model path')

    # 数据集中的模型索引
    parser.add_argument('--idx', type=int, default=0, help='model index')

    # 数据集路径
    parser.add_argument('--dataset', type=str, default=dataset_path, help='dataset path')

    # 类别选择
    parser.add_argument('--class_choice', type=str, default=class_choice, help='class choice')

    # 解析命令行参数
    opt = parser.parse_args()
    print(opt)

    # 加载 ShapeNet 数据集
    d = ShapeNetDataset(
        root=opt.dataset,
        class_choice=[opt.class_choice],
        split='test',
        data_augmentation=False)

    # 获取指定索引的点云数据和标签
    idx = opt.idx
    print("model %d/%d" % (idx, len(d)))
    point, seg = d[idx]
    print(point.size(), seg.size())
    point_np = point.numpy()

    # 获取颜色映射，用于显示点云的颜色
    cmap = plt.cm.get_cmap("hsv", 10)
    cmap = np.array([cmap(i) for i in range(10)])[:, :3]
    gt = cmap[seg.numpy() - 1, :]

    # 加载预训练模型
    state_dict = torch.load(opt.model)

    # 初始化分类器
    classifier = PointNetDenseCls(k=state_dict['conv4.weight'].size()[0])

    # 加载模型参数
    classifier.load_state_dict(state_dict)

    # 切换模型为评估模式
    classifier.eval()

    # 检查是否有可用的 GPU，如果没有则使用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用设备: ", device)

    # 将点云数据转换为网络输入格式
    point = point.transpose(1, 0).contiguous()
    point = Variable(point.view(1, point.size()[0], point.size()[1]))

    # 将点云数据移动到指定设备（GPU/CPU）
    point = point.to(device)
    classifier = classifier.to(device)

    # 执行前向传播，获取预测结果
    pred, _, _ = classifier(point)

    # 获取预测类别
    pred_choice = pred.data.max(2)[1]
    print(pred_choice)

    # 获取预测结果的颜色
    pred_color = cmap[pred_choice.cpu().numpy()[0], :]

    # 显示点云，实际标签与预测结果的比较
    showpoints(point_np, gt, pred_color)

# 主函数入口
if __name__ == '__main__':
    main()
