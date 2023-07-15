import torch
import torchvision.models as models

# 这里的backbone是一个nn.Module对象，传入ResNet-50， 我们需要修改其forward函数

class Backbone(torch.nn.Module):
    """
    ResNet-50的backbone
    @return: list，包含4个元素，每个元素是一个尺度下的feature map 
    """
    def __init__(
            self, 
            model = "resnet50",
            freeze_backbone = True
        ):
        """
        @param model: str，表示使用的backbone模型，目前只支持resnet50和resnet101
        @param freeze_backbone: bool，表示是否冻结backbone的参数
        """
        super().__init__()
        # 这里可以选择使用预训练权重，也可以不使用（也可以改成其它预训练模型）
        if model == "resnet50":
            self.resnet = models.resnet50(pretrained=True)
        elif model == "resnet101":
            self.resnet = models.resnet101(pretrained=True) 
        else:
            raise NotImplementedError("目前只支持resnet50和resnet101")

        # 冻结backbone的参数
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """
        冻结backbone的参数，即不对backbone的参数进行更新
        """
        for param in self.resnet.parameters():
            param.requires_grad = False
            
    
    def forward(self, x):
        """
        Backbone的forward函数，返回一个list，包含4个元素，每个元素是一个tensor
        @param x: 输入的tensor [batch_size, 3, h, w]
        @return: list，包含4个元素，每个元素是一个tensor 
        @note: 输出的4个tensor的shape分别是：
            [batch_size, 256, h/4, w/4]
            [batch_size, 512, h/8, w/8]
            [batch_size, 1024, h/16, w/16]
            [batch_size, 2048, h/32, w/32]
        @example:
            >>> backbone = Backbone()
            >>> x = torch.randn(1, 3, 512, 512)
            >>> output = backbone(x)
            >>> print(output[0].shape)
            >>> print(output[1].shape)
            >>> print(output[2].shape)
            >>> print(output[3].shape)
            torch.Size([1, 256, 128, 128])
            torch.Size([1, 512, 64, 64])
            torch.Size([1, 1024, 32, 32])
            torch.Size([1, 2048, 16, 16])
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        output = []

        for i in range(4):
            x = getattr(self.resnet, f'layer{i+1}')(x)
            output.append(x)

        return output
    