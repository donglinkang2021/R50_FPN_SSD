from matplotlib import pyplot as plt
import numpy as np

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

    return img