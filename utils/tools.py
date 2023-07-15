import time
import numpy as np

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

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
    
class Accumulator:
    """
    在`num_epochs`个迭代周期后，自动记录`metric`的平均值。
    n是metric累加的样本数。
    """
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]