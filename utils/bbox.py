import torch
import matplotlib.pyplot as plt

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
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """
    从（中心，宽度，高度）转换到（左上，右下） (cx, cy, w, h) -> (x1, y1, x2, y2)
    @param boxes: 边界框张量，形状为(N, 4)
    @return: 边界框张量，形状为(N, 4)
    @note: 该函数的逆函数是box_corner_to_center
    @example:
        >>> boxes = torch.tensor([[1, 1, 2, 2], [2, 3, 2, 2]], dtype=torch.float32)
        >>> box_center_to_corner(boxes)
        tensor([[0., 0., 2., 2.],
                [1., 2., 3., 4.]])
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def bbox_to_rect(bbox, color):
    """
    将边界框(左上x, 左上y, 右下x, 右下y)格式转换为matplotlib格式：
    ((左上x, 左上y), 宽, 高)
    @param bbox: 边界框
    @param color: 颜色
    @return: plt.Rectangle 
    """
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """
    显示所有边界框
    @param axes: matplotlib的坐标轴
    @param bboxes: 边界框列表，形状为(N, 4) torch.Tensor
    @param labels: 标签列表，形状为(N, 1)
    @param colors: 颜色列表，默认是五种颜色
    @return: None
    """

    def _make_list(obj, default_values=None):
        """
        将对象转换为列表
        @param obj: 对象
        @param default_values: 默认值
        @return: 列表
        """
        if obj is None: # 如果对象为空
            obj = default_values
        elif not isinstance(obj, (list, tuple)): # 如果对象不是列表或元组
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color) # detach()用于将张量从计算图中分离出来
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(
                rect.xy[0], rect.xy[1], labels[i],
                va='center', ha='center', 
                fontsize=9, color=text_color,
                bbox=dict(facecolor=color, lw=0)
            )


def multibox_prior(data, sizes, ratios):
    """
    生成以每个像素为中心具有不同形状的锚框
    @param data: 输入数据
    @param sizes: 缩放比列表
    @param ratios: 长宽比列表
    @return: 锚框张量
    @note: 生成的锚框张量的形状为(1, height * width * (len(sizes) + len(ratios) - 1), 4)
    @example:
        >>> X = torch.rand(size=(1, 3, 4, 5))
        >>> Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
        >>> Y.shape
        torch.Size([1, 100, 4])
    """
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

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
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 处理矩形输入
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def box_iou(boxes1, boxes2):
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


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """
    将最接近的真实边界框分配给锚框
    @param ground_truth: 真实边界框，形状为(num_gt_boxes, 4)
    @param anchors: 锚框，形状为(num_anchors, 4)
    @param device: 设备
    @param iou_threshold: 交并比阈值
    @return: anchors_bbox_map: 每个锚框分配的真实边界框的张量，形状为(num_anchors,)
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1) # dim=1表示沿着第1维度操作，即每一行取最大值和列索引
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1) # 取出大于阈值的列索引
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j

    # 为每个真实边界框分配锚框，每分配一个取消掉该锚框和真实边界框的IoU所在的行和列（即置为-1）
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard

    return anchors_bbox_map


def offset_boxes(anchors, assigned_bbox, eps=1e-6):
    r"""
    对锚框偏移量的转换
    @param anchors: 锚框，形状为(num_anchors, 4)
    @param assigned_bbox: 分配的真实边界框，形状为(num_anchors,4)
    @param eps: 防止零除的小常数
    @return: 偏移量，形状为(num_anchors,4)
    @note: 这里我们的计算方法为：
    - 偏移量_xy = 10 * (真实边界框的中心 - 锚框的中心) / 锚框的宽高
    - 偏移量_wh = 5 * log(真实边界框的宽高 / 锚框的宽高)
    - 偏移量 = (偏移量_xy,偏移量_wh)
    
    而实际偏移量的计算公式为：
    
    $$
    \begin{aligned}
    g_x = (x - x_a) / w_a, \\
    g_y = (y - y_a) / h_a, \\
    g_w = \log((w + \epsilon) / w_a), \\
    g_h = \log((h + \epsilon) / h_a),
    \end{aligned}
    $$
    """
    c_anchor = box_corner_to_center(anchors)
    c_assigned_bbox = box_corner_to_center(assigned_bbox)

    offset_xy = 10 * (c_assigned_bbox[:, :2] - c_anchor[:, :2]) / c_anchor[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bbox[:, 2:] / c_anchor[:, 2:])

    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset


def multibox_target(anchors, labels):
    """
    使用真实边界框标记锚框
    @param anchors: 锚框，形状为(num_anchors, 4)
    @param labels: 标签，形状为(batch_size, num_labels, 5) 这里的5表示(class, xmin, ymin, xmax, ymax)
    @return: (bbox_offset, bbox_mask, cls_labels)
    @note: 
    - bbox_offset形状为(batch_size, num_anchors * 4)
    - bbox_mask形状为(batch_size, num_anchors * 4)
    - cls_labels形状为(batch_size, num_anchors)
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        # 获取一个图像的标签，形状为(1, 5)
        label = labels[i, :, :]
        # 为每个锚框分配真实边界框
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        # 为没有分配真实边界框的锚框分配背景类，即0类，具体做法是将其xmin、ymin、xmax、ymax坐标设为0
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，我们标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
        
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


def offset_inverse(anchors, offset_preds):
    """
    根据带有预测偏移量的锚框来预测边界框
    @param anchors: 锚框，形状为(num_anchors, 4) 这里的4表示(xmin, ymin, xmax, ymax)
    @param offset_preds: 预测偏移量，形状为(num_anchors, 4) 这里的4表示(delta_x, delta_y, delta_w, delta_h)
    @return: 预测边界框，形状为(num_anchors, 4)
    @note:
    - 预测边界框的4个坐标分别为(xmin, ymin, xmax, ymax)
    - 预测边界框的4个坐标值都是相对于图像宽高的比例
    - 逆函数为`offset_boxes`
    """
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
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
        iou = box_iou(
            boxes[i, :].reshape(-1, 4),
            boxes[B[1:], :].reshape(-1, 4)
        ).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """
    使用非极大值抑制来预测边界框
    @param cls_probs: 预测类别的概率，形状为(batch_size, num_classes, num_anchors)
    @param offset_preds: 预测边界框的偏移量，形状为(batch_size, num_anchors, 4)
    @param anchors: 锚框，形状为(1, num_anchors, 4)
    @param nms_threshold: 非极大值抑制的阈值
    @param pos_threshold: 预测类别的概率的阈值
    @return: 预测边界框，形状为(batch_size, num_anchors, 6) 这里的6表示(class_id, confidence, xmin, ymin, xmax, ymax)
    """

    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]

    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1

        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]

        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((
            class_id.unsqueeze(1),
            conf.unsqueeze(1),
            predicted_bb
        ), dim=1)
        
        out.append(pred_info)
        
    return torch.stack(out)