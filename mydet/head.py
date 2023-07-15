import torch
from mydet.anchor import AnchorGenerator
from typing import Dict, List, Optional, Sequence, Tuple

class Head(torch.nn.Module):
    """
    使用SSDHead作为head
    @return: anchors, cls_preds, bbox_preds
    """
    def __init__(
            self,
            num_classes = 21,
            in_channels = [256, 256, 256, 256, 256],
            stacked_convs: int = 0,
            sizes = [[0.2 , 0.272], 
                     [0.37, 0.447], 
                     [0.54, 0.619], 
                     [0.71, 0.79 ], 
                     [0.88, 0.961]],
            ratios = [[1, 2, 0.5]] * 5,
            bbox_dim = 4,
        ):
        """
        @param num_classes: int，表示类别数，不包括背景
        @param in_channels: list，包含5个元素，每个元素是一个int，
        表示neck输出的5个tensor的通道数
        @param stacked_convs: int，表示head中的卷积层数
        @param sizes: list，包含5个元素，每个元素是一个list，
        表示该尺度特征图上的anchor的缩放比例
        @param ratios: list，包含5个元素，每个元素是一个list，表示该尺度特征图上的anchor的宽高比例
        @param bbox_dim: int，表示bbox的维度（有人也命名为loc_dim），一般是4
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.sizes = sizes
        self.ratios = ratios
        self.bbox_dim = bbox_dim

        self.cls_convs = torch.nn.ModuleList()
        self.reg_convs = torch.nn.ModuleList()

        # 构建每个尺度上的锚框生成器
        self.anchor_generators = []

        for in_channel, size, ratio in zip(in_channels, sizes, ratios):
            # 每个尺度上的一个像素点对应anchor的数量
            num_anchors = len(size) + len(ratio) - 1

            # 构建head中的卷积层
            bbox_layers = []
            cls_layers = []
            # 先构建前面卷积层
            for i in range(stacked_convs):
                bbox_layers.append(
                    torch.nn.Conv2d(in_channel, in_channel, 
                                    kernel_size=3, padding=1)
                )
                cls_layers.append(
                    torch.nn.Conv2d(in_channel, in_channel, 
                                    kernel_size=3, padding=1)
                )
                bbox_layers.append(torch.nn.ReLU())
                cls_layers.append(torch.nn.ReLU())

            # 构建最后的分类和回归层
            bbox_layers.append(
                torch.nn.Conv2d(in_channel, num_anchors * bbox_dim, kernel_size=3, padding=1)
            )
            cls_layers.append(
                torch.nn.Conv2d(in_channel, num_anchors * (num_classes+1), kernel_size=3, padding=1)
            )

            # 构建该尺度上的锚框生成器
            self.anchor_generators.append(
                AnchorGenerator(size, ratio)
            )
        
            # 添加该尺度上的卷积层添加到cls_convs和reg_convs中
            self.cls_convs.append(torch.nn.Sequential(*cls_layers))
            self.reg_convs.append(torch.nn.Sequential(*bbox_layers))

    def flatten_pred(self, pred):
        """
        将预测结果变换成二维数组
        @param pred: 输出
        @return: 二维数组
        @note: (批量大小, 通道数, 高, 宽) -> (批量大小, 高, 宽, 通道数) -> (批量大小, 高 * 宽 * 通道数)
        """
        return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

    def concat_preds(self, preds):
        """
        将多尺度的预测结果拼接
        @param preds: 预测结果列表
        @return: 拼接后的预测结果
        @note: preds中的每个元素形状为(批量大小, \sum_i{高_i * 宽_i * 通道数_i}) 其中i表示第i个尺度
        """
        return torch.cat([self.flatten_pred(p) for p in preds], dim=1)

    def forward(self, x): 
        """
        在head中前向传播上游的neck输出的特征，生成anchors，得到head的输出cls_preds和bbox_preds
        @param x: list，包含5个元素，每个元素是一个tensor，表示neck输出的5个tensor
        @return: anchors, cls_preds, bbox_preds
        @note: `cls_preds`和`bbox_preds`的每个元素都是一个tensor，
        
        拼接之前：
        - `cls_preds`的shape是`(batch_size, num_anchors * (num_classes+1), h, w)`
        - `bbox_preds`的shape是`(batch_size, num_anchors * bbox_dim, h, w)`
        
        拼接之后：
        - `cls_preds`的shape是`(batch_size, \sum_i{h_i * w_i * num_anchors_i * (num_classes+1)})`
        - `bbox_preds`的shape是`(batch_size, \sum_i{h_i * w_i * num_anchors_i * bbox_dim})`
        
        最后输出的维度：

        - `cls_preds`的shape是`(batch_size, 总的anchor数量, num_classes+1)`
        - `bbox_preds`的shape是`(batch_size, 总的anchor数量 * bbox_dim)`
        - `anchors`形状为(batch_size, 总的anchor数量, 4)

        这里的`i`表示第`i`个尺度，`h_i`和`w_i`表示第`i`个尺度上的特征图的高和宽，
        `num_anchors_i`表示第`i`个尺度上的一个像素点对应anchor的数量，
        `num_classes`表示类别数，不包括背景，
        `bbox_dim`表示bbox的维度（有人也命名为`loc_dim`），一般是4，

        另外 总的anchor数量 = \sum_i{h_i * w_i * num_anchors_i}
        """
        cls_preds = [] # 用于存储每个尺度上的分类预测结果
        bbox_preds = [] # 用于存储每个尺度上的回归预测结果
        anchors = [] # 用于存储每个尺度上的锚框
        for feature, bbox_layer, cls_layer, anchor_generator in zip(x, self.reg_convs, self.cls_convs, self.anchor_generators):
            # 生成预测结果
            cls_preds.append(cls_layer(feature))
            bbox_preds.append(bbox_layer(feature))
            # 生成锚框
            anchors.append(
                anchor_generator.generate_anchors_prior(
                    feature.shape[2],
                    feature.shape[3], 
                    feature.device
                )
            )

        # 将锚框变换维度
        anchors = torch.cat(anchors, dim=1)

        # 将预测结果变换维度
        cls_preds = self.concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes+1)
        bbox_preds = self.concat_preds(bbox_preds)

        return anchors, cls_preds, bbox_preds