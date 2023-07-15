import torch

class Neck(torch.nn.Module):
    """
    使用FPN作为neck
    @return: list，包含5个元素，每个元素是一个tensor，表示neck输出的5个尺度下的特征图
    """
    def __init__(
            self, 
            in_channels_list = [256, 512, 1024, 2048],
            out_channels = 256,
            num_outs = 5
        ):
        """
        @param in_channels_list: list，包含4个元素，每个元素是一个int，表示backbone输出的4个tensor的通道数
        @param out_channels: int，表示neck输出的tensor的通道数
        """
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_outs = num_outs

        self.lateral_convs = torch.nn.ModuleList() # lateral convs （侧面卷积）
        self.fpn_convs = torch.nn.ModuleList() # fpn convs （FPN卷积）
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.fpn_convs.append(
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
        

    def forward(self, x):
        """
        @param x: list，包含4个元素，每个元素是一个tensor，表示backbone输出的4个tensor
        @return: list，包含5个元素，每个元素是一个tensor，表示neck输出的5个tensor
        @note: 输出的5个tensor的shape分别是：
            [batch_size, out_channels, h/4, w/4]
            [batch_size, out_channels, h/8, w/8]
            [batch_size, out_channels, h/16, w/16]
            [batch_size, out_channels, h/32, w/32]
            [batch_size, out_channels, h/64, w/64]
        @example:
            >>> import torch
            >>> from mydet.neck import Neck
            >>> from mydet.backbone import Backbone
            >>> neck = Neck()
            >>> backbone = Backbone()
            >>> x = backbone(torch.randn(1, 3, 512, 512))
            >>> print("len(x):", len(x))
            >>> for i in range(len(x)):
            >>>     print("x[{}].shape:".format(i), x[i].shape)
            >>> y = neck(x)
            >>> print("len(y):", len(y))
            >>> for i in range(len(y)):
            >>>     print("y[{}].shape:".format(i), y[i].shape)
            len(x): 4
            x[0].shape: torch.Size([1, 256, 128, 128])
            x[1].shape: torch.Size([1, 512, 64, 64])
            x[2].shape: torch.Size([1, 1024, 32, 32])
            x[3].shape: torch.Size([1, 2048, 16, 16])
            len(y): 5
            y[0].shape: torch.Size([1, 256, 128, 128])
            y[1].shape: torch.Size([1, 256, 64, 64])
            y[2].shape: torch.Size([1, 256, 32, 32])
            y[3].shape: torch.Size([1, 256, 16, 16])
            y[4].shape: torch.Size([1, 256, 8, 8])
        """

        # 侧面卷积传播，将backbone的输出通道数经过1x1kernal统一转换为out_channels
        backbone_laterals = [layer_conv(backbone_output) for layer_conv, backbone_output in zip(self.lateral_convs, x)]
        
        # 从后往前遍历 Top-Down Pathway
        for i in range(len(backbone_laterals)-1, 0, -1):
            backbone_laterals[i-1] += torch.nn.functional.interpolate(backbone_laterals[i], scale_factor=2, mode='nearest')

        # FPN卷积传播，将backbone的输出通道数经过3x3kernal统一转换为out_channels
        fpn_outputs = [fpn_conv(backbone_lateral) for fpn_conv, backbone_lateral in zip(self.fpn_convs, backbone_laterals)]

        # 将超出num_outs个tensor输出全部为顶层tensor通道宽高的1/2
        for i in range(len(fpn_outputs), self.num_outs):
            fpn_outputs.append(torch.nn.functional.max_pool2d(fpn_outputs[-1], kernel_size=1, stride=2))

        return fpn_outputs

