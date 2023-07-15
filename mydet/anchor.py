import torch

class AnchorGenerator:
    """
    生成锚框的类
    @param size: 缩放比列表
    @param ratio: 长宽比列表
    @note: 生成的锚框张量的形状为(1, height * width * (len(sizes) + len(ratios) - 1), 4)
    """
    def __init__(
            self,  
            sizes,
            ratios,
        ):
        self.sizes = sizes
        self.ratios = ratios

    def generate_anchors_prior(
            self,
            fmap_h,
            fmap_w, 
            device = torch.device('cpu')
        ):
        """
        生成以每个像素为中心具有不同形状的锚框
        @param fmap_h: 特征图的高度
        @param fmap_w: 特征图的宽度
        @param device: torch.device，表示锚框张量的设备
        @return: 锚框张量
        @ref: 参考了d2l中的代码
        @note: 生成的锚框张量的形状为(1, fmap_h * fmap_w * (len(sizes) + len(ratios) - 1), 4)
        其中的每个锚框的坐标为(xmin, ymin, xmax, ymax)，坐标值为相对于原图的比例
        @example:
            >>> X = torch.rand(size=(1, 3, 4, 5))
            >>> Y = AnchorGenerator(
            >>>         sizes = [0.75, 0.5, 0.25],
            >>>         ratios = [1, 2, 0.5],
            >>>     ).generate_anchors_prior(
            >>>         fmap_h = X.shape[2],
            >>>         fmap_w = X.shape[3],
            >>>         device = X.device
            >>>     )
            >>> print(Y.shape)
            torch.Size([1, 100, 4])
        """
        in_height, in_width = fmap_h, fmap_w
        device = device
        boxes_per_pixel = len(self.sizes) + len(self.ratios) - 1
        size_tensor = torch.tensor(self.sizes, device=device)
        ratio_tensor = torch.tensor(self.ratios, device=device)

        # 为了将锚点移动到像素的中心，需要设置偏移量。
        # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
        offset_h, offset_w = 0.5, 0.5
        steps_h = 1.0 / in_height  # 在y轴上缩放步长
        steps_w = 1.0 / in_width  # 在x轴上缩放步长

        # 生成锚框的所有中心点
        center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
        center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
        shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
        shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

        # 生成“boxes_per_pixel”个高和宽，
        # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
        w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                    self.sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                    * in_height / in_width  # 处理矩形输入
        h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                    self.sizes[0] / torch.sqrt(ratio_tensor[1:])))
        
        # 除以2来获得半高和半宽
        anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                            in_height * in_width, 1) / 2

        # 每个中心点都将有“boxes_per_pixel”个锚框，
        # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
        out_grid = torch.stack(
            [shift_x, shift_y, shift_x, shift_y],dim=1
        ).repeat_interleave(boxes_per_pixel, dim=0)

        output = out_grid + anchor_manipulations
        return output.unsqueeze(0)
    
