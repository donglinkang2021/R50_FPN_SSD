import os
import pandas as pd
import json
import torch
import torchvision
import torchvision.transforms as transforms


# 训练数据及标注
train_data_dir = "./dl_detection/train/"
train_anno_fname = "./dl_detection/annotations/train.json"

# 测试数据
test_data_dir = "./dl_detection/test/"

# 读取标注文件
def read_anno_file(fname):
    """
    读取标注文件
    @param fname: 标注文件名
    @return: 一个dict，具体介绍在`0查看数据集`中
    """
    anno = open(fname, 'rt', encoding='UTF-8')
    anno = json.load(anno)
    return anno

class Anno:
    """
    定义一个data class，用于存储标注信息
    """
    def __init__(self):
        self.anno = read_anno_file(train_anno_fname)

    def get_image_info(self, idx):
        """
        获取id为idx的图片信息
        @param idx: 图片索引
        @return: 图片信息
        """
        for image in self.anno['images']:
            if image['id'] == idx:
                return image
        
anno = Anno() # 只需要读取一次标注文件


def target_transform(target):
    """
    @param target: 标注信息 一个list，其中每个元素为一个dict，包含了一个bbox的信息
    @return: 转换后的标注信息 一个tensor，其中每一行为一个bbox的信息
    @note:
    target 的格式为：
    [
        {
            'iscrowd': 0,
            'image_id': 0,
            'bbox': [x_min, y_min, width, height],
            'category_id': 0,
            'id': 0
        },
        ...
    ]

    为了和之前写的包所匹配起来，我们需要将其转换为一个tensor，其格式为：
    [
        [category_id, x_min, y_min, x_max, y_max],
        ...
    ]
    """
    image_id = target[0]["image_id"]
    
    # 获取图片信息
    image_info = anno.get_image_info(image_id)
    h, w = image_info["height"], image_info["width"]

    edge = torch.tensor([w, h, w, h])

    target = torch.tensor(
        [[t["category_id"], 
        t["bbox"][0], t["bbox"][1], 
        t["bbox"][0]+t["bbox"][2], 
        t["bbox"][1]+t["bbox"][3]] for t in target])
    
    target[:, 1:] = target[:, 1:] / edge

    rows = target.shape[0]

    # 将target中的bbox个数扩展为50个，因为图片中最多有48个bbox
    if target.shape[0] < 50:
        target = torch.cat(
            [
                target,
                torch.zeros(50 - target.shape[0], 5)
            ]
        )
        # 将新加入的bbox的cls设置为-1
        # target[rows:, 0] = -1
    else:
        target = target[:50]
    
    return target


def read_Cocodata_train(is_transform = True):
    """
    读取COCO数据集
    @param is_transform: 是否对数据集进行转换
    @return: Cocodata_train 数据集
    @note: 该数据集包含了训练数据集中的所有图片及其标注，在测试的时候可以不transform直接查看数据集中的图片
    其中target的第一个维度为图片中bbox的个数，最多为50个，如果图片中bbox的个数小于50个，则用0进行填充bbox，用0进行填充cls
    cls的label为0-21，bbox的坐标为相对坐标，即x_min, y_min, x_max, y_max的值为0-1
    """

    if is_transform == False:
        return torchvision.datasets.CocoDetection(
            root = train_data_dir, 
            annFile = train_anno_fname
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((256, 256))
    ])

    # 读取训练数据集
    Cocodata_train = torchvision.datasets.CocoDetection(
        root = train_data_dir, 
        annFile = train_anno_fname, 
        transform = transform,
        target_transform = target_transform 
    )

    return Cocodata_train

def show_sample(Cocodata_train, idx):
    """
    显示数据集中的样本（使用plt画出bbox和cls）
    @param Cocodata_train: 没有transform的数据集
    @param idx: 样本索引
    @return: None
    """
    img, anno = Cocodata_train[idx]
    labels = []
    bboxs = []
    for elem in anno:
        labels.append(elem['category_id'])
        bboxs.append(elem['bbox'])
    from utils.draw import draw_bboxs
    draw_bboxs(
        img,
        bboxs,
        labels,
    )

def test(idx = 0):
    Cocodata_train = read_Cocodata_train(is_transform = False)
    show_sample(Cocodata_train, idx)


if __name__ == "__main__":
    test(0)