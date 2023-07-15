import torch
from mydet.bbox.coder import BBoxCoder
from mydet.bbox.iou_nms import bbox_nms


def bbox_filter(bbox_preds):
    """
    过滤掉bbox offset小于0的bbox
    @param bbox_preds: torch.tensor bbox预测结果，shape: [n, 6] 6: [class_id, score, x_min, y_min, x_max, y_max]
    @return: 过滤后的bbox预测结果
    """
    def condition(bbox):
        """
        只要x_min, y_min, width, height(相对比例)都大于0且小于1，
        且x_min < x_max, y_min < y_max，就认为是有效的bbox
        @param bbox: bbox信息 [x_min, y_min, width, height]
        @return: 是否满足条件
        """
        return torch.all(bbox[2:] > 0).item() and \
            torch.all(bbox[2:] < 1).item() and \
            torch.all(bbox[2:4] < bbox[4:6]).item()

    # 获取满足条件的bbox的索引
    idx = [i for i, bbox_pred in enumerate(bbox_preds) if condition(bbox_pred)]
    
    return bbox_preds[idx]

class BBoxPostProcessor:
    """
    用于评估的时候输出预测边界框
    @param nms_threshold: 非极大值抑制的阈值(0-1)，这个值越大，就允许越多的重叠
    @param neg_threshold: 预测类别为背景的概率的阈值(0-1)，这个值越大，就允许越多的背景
    @note: 评估的时候要用到这个类
    """

    def __init__(
            self, 
            nms_threshold = 0.5, 
            neg_threshold = 0.009999999,
            device = torch.device('cpu')
        ):
        """
        @param nms_threshold: 非极大值抑制的阈值
        @param neg_threshold: 预测类别的概率的阈值
        @param device: 计算设备
        """
        self.nms_threshold = nms_threshold
        self.neg_threshold = neg_threshold
        self.bbox_coder = BBoxCoder(device=device)


    def multibox_detection(
            self, 
            cls_probs, 
            offset_preds, 
            anchors
        ):
        """
        使用非极大值抑制来预测边界框
        @param cls_probs: 预测类别的概率，形状为(batch_size, num_classes, num_anchors)
        @param offset_preds: 预测边界框的偏移量，形状为(batch_size, num_anchors, 4)
        @param anchors: 锚框，形状为(1, num_anchors, 4)
        @return: 预测边界框，形状为(batch_size, num_anchors, 6) 这里的6表示(class_id, confidence, xmin, ymin, xmax, ymax)
        """

        device, batch_size = cls_probs.device, cls_probs.shape[0]
        anchors = anchors.squeeze(0)
        num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]

        out = []
        for i in range(batch_size):
            cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
            conf, class_id = torch.max(cls_prob[1:], 0)
            predicted_bb = self.bbox_coder.decode(anchors, offset_pred)

            keep = bbox_nms(predicted_bb, conf, self.nms_threshold)

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
            below_min_idx = (conf < self.neg_threshold)

            class_id[below_min_idx] = -1
            conf[below_min_idx] = 1 - conf[below_min_idx]
            pred_info = torch.cat((
                class_id.unsqueeze(1),
                conf.unsqueeze(1),
                predicted_bb
            ), dim=1)
            
            out.append(pred_info)
            
        return torch.stack(out)