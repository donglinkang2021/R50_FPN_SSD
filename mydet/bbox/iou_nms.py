import torch

def bbox_iou(boxes1, boxes2):
    """
    计算两个锚框或边界框列表中成对的交并比
    @param boxes1: 第一个锚框或边界框列表，形状为(boxes1的数量, 4)
    @param boxes2: 第二个锚框或边界框列表，形状为(boxes2的数量, 4)
    @return: 交并比矩阵，形状为(boxes1的数量,boxes2的数量)
    @note: 交并比是一个矩阵，其中第i行第j列的元素是第一个锚框或边界框列表中第i个元素与第二个锚框或边界框列表中第j个元素的交并比
    @example:
        >>> boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0],
        >>>                        [0.0, 0.0, 0.5, 0.5]])
        >>> boxes2 = torch.tensor([[1.0, 1.0, 2.0, 2.0],
        >>>                        [0.5, 0.5, 1.0, 1.0],
        >>>                        [0.0, 0.0, 0.5, 0.5]])
        >>> box_iou(boxes1, boxes2)
        tensor([[0.0000, 0.2500, 0.2500], 
                [0.0000, 0.0000, 1.0000]])
    """
    # 计算每个锚框或边界框的面积的lambda函数
    # shape: (boxes的数量, 4) -> (boxes的数量,)
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    
    areas1 = box_area(boxes1) # areas1：(boxes1的数量,)
    areas2 = box_area(boxes2) # areas2：(boxes2的数量,)

    # 找出交集左上角和交集右下角的坐标
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    # clamp(min=0)将小于0的值设为0
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0) 

    # inter_areas union_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas

    return inter_areas / union_areas

def bbox_nms(boxes, scores, iou_threshold):
    """
    对预测边界框的置信度进行排序，保留置信度高的边界框，并剔除与其高度重合的边界框
    @param boxes: 预测边界框，形状为(num_anchors, 4)
    @param scores: 预测边界框的置信度，形状为(num_anchors, )
    @param iou_threshold: 非极大值抑制的阈值
    @return: 保留预测边界框的指标, 形状为(num_anchors, )
    """
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = bbox_iou(
            boxes[i, :].reshape(-1, 4),
            boxes[B[1:], :].reshape(-1, 4)
        ).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)