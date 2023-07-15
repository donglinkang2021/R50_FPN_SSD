import numpy as np

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

def bboxListToStr(bbox_list):
    """
    @brief      将bbox列表转换为字符串
    @param      `bbox_list`     bbox列表
    @return     `bbox_str`      bbox字符串
    @note
    将bbox列表转换为字符串，其中bbox_list的格式为：`[[x_l, y_l, x_r, y_r, conf, cls], [x_l, y_l, x_r, y_r, conf, cls], ...]`
    转换后的bbox_str的格式为：`"{x_l y_l x_r y_r conf cls}{x_l y_l x_r y_r conf cls}..."`
    @example
    >>> bbox_list = [[265.362, 70.297, 335.99100000000004, 165.14999999999998, 0.6416, 6], [594.026, 87.9, 640.0, 166.44600000000003, 0.2053, 6]]
    >>> bbox_str = bboxListToStr(bbox_list)
    >>> print(bbox_str)
    {265.362 70.297 335.99100000000004 165.14999999999998 0.6416 6}{594.026 87.9 640.0 166.44600000000003 0.2053 6}
    """
    bbox_str = ""
    for bbox in bbox_list:
        bbox_str += "{" + " ".join([
            str(bbox[0]),
            str(bbox[1]),
            str(bbox[2]),
            str(bbox[3]),
            str(bbox[4]),
            str(int(bbox[5]))
        ]) + "}"
    return bbox_str

def test1():
    bbox_str = "{265.362 70.297 335.99100000000004 165.14999999999998 0.6416 6}{594.026 87.9 640.0 166.44600000000003 0.2053 6}"
    bbox_list = bboxStrToList(bbox_str)
    print(bbox_list)
    bbox_np_list = np.array(bbox_list) # 将bbox_list转换为numpy数组
    print(bbox_np_list)
    print(bbox_np_list.shape)
    print(bbox_np_list.dtype)

def test2():
    bbox_list = [[265.362, 70.297, 335.99100000000004, 165.14999999999998, 0.6416, 6], [594.026, 87.9, 640.0, 166.44600000000003, 0.2053, 6]]
    bbox_str = bboxListToStr(bbox_list)
    print(bbox_str)

if __name__ == "__main__":
    # test1()
    test2()

