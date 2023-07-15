import torch
from mydet.bbox.coder import BBoxCoder
from mydet.bbox.iou_nms import bbox_iou

class BBoxAssigner:
    """
    为锚框分配真实标签的类
    @param device: 设备
    @param iou_threshold: 交并比阈值
    @note: 训练的时候要用到这个类
    """

    def __init__(
            self,
            device,
            iou_threshold = 0.5,
        ):
        self.device = device
        self.iou_threshold = iou_threshold
        self.bbox_coder = BBoxCoder(device=device)

    def assign_anchor_to_bbox(self, ground_truth, anchors):
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
        jaccard = bbox_iou(anchors, ground_truth)
        # 对于每个锚框，分配的真实边界框的张量
        anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                    device=self.device)
        # 根据阈值，决定是否分配真实边界框
        max_ious, indices = torch.max(jaccard, dim=1) # dim=1表示沿着第1维度操作，即每一行取最大值和列索引
        anc_i = torch.nonzero(max_ious >= self.iou_threshold).reshape(-1) # 取出大于阈值的列索引
        box_j = indices[max_ious >= self.iou_threshold]
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


    def multibox_target(self, anchors, labels):
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
            anchors_bbox_map = self.assign_anchor_to_bbox(label[:, 1:], anchors)
            # 为没有分配真实边界框的锚框分配背景类，即0类，具体做法是将其xmin、ymin、xmax、ymax坐标设为0
            bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
            # 将类标签和分配的边界框坐标初始化为零
            class_labels = torch.zeros(num_anchors, 
                                       dtype=torch.long,
                                       device=device)
            assigned_bb = torch.zeros((num_anchors, 4), 
                                      dtype=torch.float32,
                                      device=device)
            # 使用真实边界框来标记锚框的类别。
            # 如果一个锚框没有被分配，我们标记其为背景（值为零）
            indices_true = torch.nonzero(anchors_bbox_map >= 0)
            bb_idx = anchors_bbox_map[indices_true]
            class_labels[indices_true] = label[bb_idx, 0].long() + 1
            assigned_bb[indices_true] = label[bb_idx, 1:]
            # 偏移量转换
            offset = self.bbox_coder.encode(anchors, assigned_bb) * bbox_mask
            batch_offset.append(offset.reshape(-1))
            batch_mask.append(bbox_mask.reshape(-1))
            batch_class_labels.append(class_labels)
            
        bbox_offset = torch.stack(batch_offset)
        bbox_mask = torch.stack(batch_mask)
        class_labels = torch.stack(batch_class_labels)

        return (bbox_offset, bbox_mask, class_labels)