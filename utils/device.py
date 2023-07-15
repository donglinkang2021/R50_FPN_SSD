import torch

def try_gpu(i=0):
    """
    尝试使用GPU，如果不可用则返回CPU
    @param i: GPU的编号
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
