import torch

def box_center_to_corner(boxes):
    """
    (center_x, center_y, w, h) -> (x1, y1, x2, y2)
    @param boxes: 边界框张量，形状为(N, 4)
    @return: 边界框张量，形状为(N, 4)
    @note: 该函数的逆函数是box_corner_to_center
    @example:
        >>> boxes = torch.tensor([[1, 1, 2, 2], [2, 3, 2, 2]], dtype=torch.float32)
        >>> box_center_to_corner(boxes)
        tensor([[0., 0., 2., 2.],
                [1., 2., 3., 4.]])
    """
    return torch.cat((
        boxes[:, :2] - boxes[:, 2:] / 2,
        boxes[:, :2] + boxes[:, 2:] / 2
    ), axis=-1)

def box_corner_to_center(boxes):
    """
    从（左上，右下）转换到（中心，宽度，高度） (x1, y1, x2, y2) -> (cx, cy, w, h)
    @param boxes: 边界框张量，形状为(N, 4)
    @return: 边界框张量，形状为(N, 4)
    @note: 该函数的逆函数是box_center_to_corner
    @example:
        >>> boxes = torch.tensor([[0, 0, 2, 2], [1, 2, 3, 4]], dtype=torch.float32)
        >>> box_corner_to_center(boxes)
        tensor([[1., 1., 2., 2.],
                [2., 3., 2., 2.]])
    """
    return torch.cat((
        (boxes[:, :2] + boxes[:, 2:]) / 2,
        boxes[:, 2:] - boxes[:, :2]
    ), axis=-1)

class BBoxCoder:
    """
    边界框编码器
    """
    def __init__(
            self, 
            eps = 1e-6,
            means = (0., 0., 0., 0.),
            stds = (0.1, 0.1, 0.2, 0.2),
            device = torch.device('cpu')
        ):
        """
        @param eps: 防止零除的小常数
        @param means: 均值
        @param stds: 标准差
        @param device: 设备
        """
        self.eps = eps
        self.means = torch.tensor(means, device=device)
        self.stds = torch.tensor(stds, device=device)

    def encode(self, anchors, assigned_bbox):
        """
        对锚框偏移量的转换，即通过真实边界框和锚框计算锚框的偏移量
        @param anchors: 锚框，形状为(num_anchors, 4)
        @param assigned_bbox: 分配的真实边界框，形状为(num_anchors,4)
        @return: 偏移量，形状为(num_anchors,4)
        @note: 这里我们的计算方法为：
        - 偏移量_xy = (真实边界框的中心 - 锚框的中心) / 锚框的宽高
        - 偏移量_wh = log(真实边界框的宽高 / 锚框的宽高)
        - 偏移量 = ((偏移量_xy,偏移量_wh) - 均值) / 标准差
        """
        c_anchor = box_corner_to_center(anchors)
        c_assigned_bbox = box_corner_to_center(assigned_bbox)
        delta_xy = (c_assigned_bbox[:, :2] - c_anchor[:, :2]) / c_anchor[:, 2:]
        delta_wh = torch.log(self.eps + c_assigned_bbox[:, 2:] / c_anchor[:, 2:])
        return (torch.cat([delta_xy, delta_wh], axis=1) - self.means) / self.stds

    def decode(self, anchors, offset_preds):
        """
        根据带有预测偏移量的锚框来预测边界框，即输入锚框和预测偏移量，输出预测边界框
        @param anchors: 锚框，形状为(num_anchors, 4) 这里的4表示(xmin, ymin, xmax, ymax)
        @param offset_preds: 预测偏移量，形状为(num_anchors, 4) 这里的4表示(delta_x, delta_y, delta_w, delta_h)
        @return: 预测边界框，形状为(num_anchors, 4)
        @note:
        - 预测边界框的4个坐标分别为(xmin, ymin, xmax, ymax)
        - 预测边界框的4个坐标值都是相对于图像宽高的比例
        - 逆函数为`offset_boxes`
        """
        c_anchor = box_corner_to_center(anchors)
        origin_pred = offset_preds * self.stds + self.means
        pred_xy = origin_pred[:, :2] * c_anchor[:, 2:] + c_anchor[:, :2]
        pred_wh = torch.exp(origin_pred[:, 2:]) * c_anchor[:, 2:]
        pred_bbox = torch.cat([pred_xy, pred_wh], axis=1)
        predicted_bbox = box_center_to_corner(pred_bbox)
        return predicted_bbox

# boxes = torch.tensor(
#     [[1, 1, 2, 2], 
#      [2, 3, 2, 2]], 
#      dtype=torch.float32)
# print(box_center_to_corner(boxes))
# print(box_corner_to_center(box_center_to_corner(boxes)))