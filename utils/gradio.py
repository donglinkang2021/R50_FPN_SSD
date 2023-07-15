# 直观看一下选定锚框参数后生成的效果
from matplotlib import pyplot as plt
from utils.bbox import *
import numpy as np
import torch
import gradio as gr

def display_anchors(img, fmap_w, fmap_h, sizes=[0.15], ratios=[1, 2, 0.5]):
    """
    显示不同尺寸的锚框
    @param img: 图像
    @param fmap_w: 特征图宽度
    @param fmap_h: 特征图高度
    @param sizes: 缩放比例列表
    @param ratios: 长宽比列表
    @return: None
    """
    w, h = img.shape[1], img.shape[0]
    plt.figure(figsize=(3.5, 3.5))
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 5, fmap_h, fmap_w)) # m, c, h, w
    anchors = multibox_prior(fmap, sizes, ratios)
    bbox_scale = torch.tensor((w, h, w, h))
    show_bboxes(plt.imshow(img).axes, anchors[0] * bbox_scale)


def process(img, fmap_w=4, fmap_h=4, sizes=[0.15], ratios=[1, 2, 0.5]):
    display_anchors(img, fmap_w, fmap_h, sizes, ratios)
    return plt.gcf()

def show_multi_anchors_app(
        sizes=[0.1, 0.15, 0.2, 0.4, 0.8],
        ratios=[1, 2, 0.5, 4, 0.25]
):
    """
    显示多尺度锚框的交互式应用
    @param sizes: 缩放比例列表
    @param ratios: 长宽比列表
    @return: None
    """
    with gr.Blocks() as demo:
        gr.Markdown("""
        # 多尺度锚框

        本示例展示了如何使用多尺度锚框来检测不同大小的目标。具体的代码是参考了d2l的[多尺度锚框](https://zh-v2.d2l.ai/chapter_computer-vision/multiscale-object-detection.html)一节。

        这里使用了gradio来构建一个简单的交互式界面，你可以通过调整参数来查看不同的结果。

        """)

        with gr.Row():
            img = gr.inputs.Image(label="图像")
            img_with_anchors = gr.Plot()
        
        with gr.Row():
            fmap_w = gr.inputs.Slider(minimum=1, maximum=10, default=4, step=1, label="特征图宽度fmap_w")
            fmap_h = gr.inputs.Slider(minimum=1, maximum=10, default=4, step=1, label="特征图高度fmap_h")
        with gr.Row():
            sizes = gr.inputs.CheckboxGroup(choices=sizes,
                                            label="缩放比例sizes")
            ratios = gr.inputs.CheckboxGroup(choices=ratios,
                                            label="长宽比ratios")
        
        
        run_btn = gr.Button(label="运行")
        

        img_example = gr.Examples([
            ["./image/catdog.jpg"],
            ["./image/dog.jpg"],
            ["./image/cat.jpg"],
        ], label="示例",inputs=[img], outputs=img_with_anchors)

        
        run_btn.click(
            process,
            inputs=[
                img,
                fmap_w,
                fmap_h,
                sizes,
                ratios
            ],
            outputs=img_with_anchors
        )

    return demo.launch()
