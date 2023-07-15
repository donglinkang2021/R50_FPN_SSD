import numpy as np
import matplotlib.pyplot as plt

def plt2np(
        plt
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