import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from utils.bbox import *

# region 训练模型


def cls_predictor(num_inputs, num_anchors, num_classes):
    """
    类别预测层
    @param num_inputs: 输入通道数
    @param num_anchors: 每个像素点的锚框数量
    @param num_classes: 类别数量
    @return: 类别预测层
    @note: 该层输出形状为(批量大小, num_anchors * (num_classes + 1), 高, 宽)
    """
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)


def bbox_predictor(num_inputs, num_anchors):
    """
    边界框预测层
    @param num_inputs: 输入通道数
    @param num_anchors: 每个像素点的锚框数量
    @return: 边界框预测层
    @note: 该层输出形状为(批量大小, num_anchors * 4, 高, 宽)
    """
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def flatten_pred(pred):
    """
    将预测结果变换成二维数组
    @param pred: 输出
    @return: 二维数组
    @note: (批量大小, 通道数, 高, 宽) -> (批量大小, 高, 宽, 通道数) -> (批量大小, 高 * 宽 * 通道数)
    """
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    """
    将多尺度的预测结果拼接
    @param preds: 预测结果列表
    @return: 拼接后的预测结果
    @note: preds中的每个元素形状为(批量大小, \sum_i{height_i * width_i * channels_i})
    """
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels, out_channels):
    """
    下采样模块：两个卷积层和一个最大池化层，功能是将输入特征图的高和宽减半
    @param in_channels: 输入通道数
    @param out_channels: 输出通道数
    @return: 下采样模块
    @note: 该模块输出形状为(批量大小, out_channels, 高 / 2, 宽 / 2)
    """
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    """
    基础网络模块：3个下采样模块
    @return: 基础网络模块
    @note: 该模块输出形状为(批量大小, 64, 高 / 8, 宽 / 8)
    """
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)


def get_blk(i):
    """
    获取第i个模块
    @param i: 模块编号
    @return: 第i个模块
    """
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    """
    一个块的前向传播
    @param X: 输入
    @param blk: 块
    @param size: 缩放比例列表
    @param ratio: 长宽比列表
    @param cls_predictor: 类别预测层
    @param bbox_predictor: 边界框预测层
    @return: 块的输出
    @note: 块的输出包括特征图、锚框、类别预测和边界框预测，

    - 其中特征图Y形状为(批量大小, 通道数, 高, 宽)，
    - 锚框形状为(1, 高 * 宽 * num_anchors, 4)，
    - 类别预测形状为(批量大小, num_anchors*(num_classes + 1), 高, 宽)，
    - 边界框预测形状为(批量大小, num_anchors*4, 高, 宽)
    """
    Y = blk(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


class TinySSD(nn.Module):
    """
    TinySSD模型
    @param num_classes: 类别数
    @param sizes: 缩放比例列表
    @param ratios: 长宽比列表
    @param kwargs: 其他参数
    """

    def __init__(self, num_classes, sizes, ratios, **kwargs):
        """
        TinySSD模型
        @param num_classes: 类别数
        @param sizes: 缩放比例列表
        @param ratios: 长宽比列表
        @param kwargs: 其他参数
        """
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.sizes = sizes
        self.ratios = ratios
        num_anchors = len(sizes[0]) + len(ratios[0]) - 1
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        """
        前向传播
        @param X: 输入
        @return: 输出
        @note: 输出包括锚框、类别预测和边界框预测，
        - `anchors`形状为(批量大小, num_anchors, 4)
        - `cls_preds`形状为(批量大小, num_anchors, (num_classes + 1))
        - `bbox_preds`形状为(批量大小, num_anchors*4)
        """
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), self.sizes[i], self.ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


# endregion


# region 计算损失函数和评价函数

def calc_loss(
    cls_preds,
    cls_labels,
    bbox_preds,
    bbox_labels,
    bbox_masks
):
    """
    计算损失
    @param cls_preds: 类别预测 （批量大小，锚框数，类别数）
    @param cls_labels: 类别标签 （批量大小，锚框数）
    @param bbox_preds: 边界框预测 （批量大小，锚框数，4）
    @param bbox_labels: 边界框标签 （批量大小，锚框数，4）
    @param bbox_masks: 边界框掩码 （批量大小，锚框数，4）
    @return: 损失 （批量大小，）
    @note: 损失包括类别损失和边界框损失，其中类别损失使用的是交叉熵损失函数，边界框损失使用的是L1范数损失函数
    """

    cls_loss = nn.CrossEntropyLoss(reduction='none')  # 设置为none的意思是不对损失求平均
    bbox_loss = nn.L1Loss(reduction='none')  # 同上，是一个向量，每个元素是一个样本的损失

    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1) # 由于是dim=1 是对每一个锚框求平均
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1) # 由于是dim=1 是对每一个锚框求平均
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    """
    评估类别预测
    @param cls_preds: 类别预测 （批量大小，锚框总个数，类别数）
    @param cls_labels: 类别标签 （批量大小，锚框总个数）
    @return: 正确的总个数
    @note: 
    - 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    - （cls_preds的维度是（批量大小，锚框总个数，类别数）
    """
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    """
    评估边界框预测
    @param bbox_preds: 边界框预测 （批量大小，锚框总个数，4）
    @param bbox_labels: 边界框标签 （批量大小，锚框总个数，4）
    @param bbox_masks: 边界框掩码 （批量大小，锚框总个数，4）
    @return: 返回绝对误差之和   
    """
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


# endregion
