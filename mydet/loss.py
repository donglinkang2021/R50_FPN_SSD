import torch.nn as nn

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