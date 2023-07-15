# 将一些常用的函数放在这里，方便调用
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

categories_name = [
 'backpack',
 'cup',
 'bowl',
 'banana',
 'apple',
 'orange',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'laptop',
 'mouse',
 'keyboard',
 'cell phone',
 'book',
 'clock',
 'vase',
 'scissors',
 'hair drier',
 'toothbrush'
]


# region 读取数据
def print_dict(d):
    """
    @brief      打印字典
    @param      `d`     字典
    @return     `None`
    @note
    定义一个函数，用于打印字典中的内容，其中用到了递归，当字典中的元素是字典时，递归调用该函数

    @example
    >>> d = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}}
    >>> print_dict(d)
    a : 1;  b : 2;  c : {'d': 3, 'e': 4};  
    """
    for elem in d:
        if type(elem) == dict:
            print_dict(elem)
            print()
        else:
            print(elem, ':', d[elem], end=';  ')


# endregion

# region submit
def bboxStrToList(bbox_str):
    """
    @brief      将bbox字符串转换为列表
    @param      `bbox_str`      bbox字符串
    @return     `bbox_list`     bbox列表
    @note
    将bbox字符串转换为列表，其中bbox_str的格式为：`"{x_l y_l x_r y_r conf cls}{x_l y_l x_r y_r conf cls}..."`
    转换后的bbox_list的格式为：`[[x_l, y_l, x_r, y_r, conf, cls], [x_l, y_l, x_r, y_r, conf, cls], ...]`
    @example
    >>> bbox_str = "{265.362 70.297 335.99100000000004 165.14999999999998 0.6416 6}{594.026 87.9 640.0 166.44600000000003 0.2053 6}"
    >>> bbox_list = bboxStrToList(bbox_str)
    >>> print(bbox_list)
    [[265.362, 70.297, 335.99100000000004, 165.14999999999998, 0.6416, 6], [594.026, 87.9, 640.0, 166.44600000000003, 0.2053, 6]]
    """
    bbox_str = bbox_str.replace(" ", ',').replace("}{", "}, {")
    bbox_str_list = bbox_str.split(", ")
    bbox_list = []
    for bbox in bbox_str_list:
        x_l, y_l, x_r, y_r, conf, cls = bbox.strip("{}").split(",")
        bbox_list.append([
            float(x_l),
            float(y_l),
            float(x_r),
            float(y_r),
            float(conf),
            int(cls)
        ])
    return bbox_list

def test1():
    bbox_str = "{265.362 70.297 335.99100000000004 165.14999999999998 0.6416 6}{594.026 87.9 640.0 166.44600000000003 0.2053 6}"
    bbox_list = bboxStrToList(bbox_str)
    print(bbox_list)
    bbox_np_list = np.array(bbox_list) # 将bbox_list转换为numpy数组
    print(bbox_np_list)
    print(bbox_np_list.shape)
    print(bbox_np_list.dtype)

# endregion

# region draw

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

# endregion

# region gradio

def plt2np(
        plt, 
        # origin_weight, 
        # origin_height
    ):
    """将plt画出的图像转换为numpy数组
    @param plt: plt
    @return img: img
    """
    # 获取当前画布对象
    canvas = plt.get_current_fig_manager().canvas

    # 将画布转换为 numpy 数组
    canvas.draw()
    width, height = canvas.get_width_height()
    img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape((height, width, 3))

    # 裁剪掉多余的部分
    # cx = width // 2
    # cy = height // 2
    # x1, y1 = cx - origin_weight // 2, cy - origin_height // 2
    # x2, y2 = cx + origin_weight // 2, cy + origin_height // 2
    # img = img[y1:y2, x1:x2, :]

    return img

# endregion

if __name__ == "__main__":
    test1()