import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

categories_name = [
 'backpack', # 背包
 'cup', # 杯子
 'bowl', # 碗
 'banana', # 香蕉
 'apple', # 苹果
 'orange', # 橘子
 'chair', # 椅子
 'couch', # 沙发
 'potted plant', # 盆栽
 'bed', # 床
 'dining table', # 餐桌
 'laptop', # 笔记本电脑
 'mouse', # 鼠标
 'keyboard', # 键盘
 'cell phone', # 手机
 'book', # 书
 'clock', # 时钟
 'vase', # 花瓶
 'scissors', # 剪刀
 'hair drier', # 吹风机
 'toothbrush' # 牙刷
]


def draw_bboxs(
        img, 
        bboxs, 
        labels, 
        colors = ['b', 'g', 'r', 'm', 'c', 'y'],
        text_color='w', 
        fontsize=11,
        thickness=2
    ):
    """在图片上画出bbox
    @param img: PIL.Image类型
    @param bboxs: list of bbox [x, y, w, h]
    @param labels: list of label [0~20]
    @param colors: 锚框颜色集合
    @param text_color: 文字颜色
    @param thickness: 线条粗细
    """
    fig = plt.imshow(img)
    for idx, bbox in enumerate(bboxs):
        color = colors[idx % len(colors)]
        rect = plt.Rectangle(
            xy=(bbox[0], bbox[1]), 
            width=bbox[2], 
            height=bbox[3], 
            fill=False, 
            edgecolor=color, 
            linewidth=thickness
        )
        fig.axes.add_patch(rect)
        fig.axes.text(
            bbox[0], bbox[1], # bbox的左上角坐标
            categories_name[labels[idx]], # label
            va='center', # vertical alignment
            ha='center', # horizontal alignment
            fontsize=fontsize,
            color=text_color,
            bbox=dict(facecolor=color, lw=0)
        )

def draw_bboxs_with_conf(
        img, 
        bboxs, 
        labels, 
        confs, 
        colors = ['b', 'g', 'r', 'm', 'c', 'y'], 
        text_color='w', 
        fontsize=11,
        thickness=2
    ):
    """在图片上画出带置信度的bbox
    @param img: PIL.Image类型
    @param bboxs: list of bbox [x, y, w, h]
    @param labels: list of label [0~20]
    @param confs: list of conf [0.0~1.0]
    @param colors: 锚框颜色集合
    @param text_color: 文字颜色
    @param thickness: 线条粗细
    """
    fig = plt.imshow(img)
    for idx, bbox in enumerate(bboxs):
        color = colors[idx % len(colors)]
        rect = plt.Rectangle(
            xy=(bbox[0], bbox[1]), 
            width=bbox[2], 
            height=bbox[3], 
            fill=False, 
            edgecolor=color, 
            linewidth=thickness
        )
        fig.axes.add_patch(rect)
        fig.axes.text(
            bbox[0], bbox[1], # bbox的左上角坐标
            categories_name[labels[idx]] + ' ' + '%2d%%'%(confs[idx]*100), # label
            va='center', # vertical alignment
            ha='center', # horizontal alignment
            fontsize=fontsize,
            color=text_color,
            bbox=dict(facecolor=color, lw=0)
        )

