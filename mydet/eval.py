import torch

def cls_eval(cls_preds, cls_labels):
    """
    评估类别预测
    @param cls_preds: 类别预测 （批量大小，锚框总个数，类别数）
    @param cls_labels: 类别标签 （批量大小，锚框总个数）
    @return: 正确的总个数
    @note: 
    - 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    - （cls_preds的维度是（批量大小，锚框总个数，类别数）
    """
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    """
    评估边界框预测
    @param bbox_preds: 边界框预测 （批量大小，锚框总个数，4）
    @param bbox_labels: 边界框标签 （批量大小，锚框总个数，4）
    @param bbox_masks: 边界框掩码 （批量大小，锚框总个数，4）
    @return: 返回绝对误差之和   
    """
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
