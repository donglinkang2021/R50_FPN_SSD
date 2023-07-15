import time

def download_file():
    # 模拟下载文件
    time.sleep(0.5)

def print_progress_bar(
        iteration, total, 
        prefix='', 
        suffix='', 
        decimals=1, 
        length=100, 
        fill='█', 
        print_end='\r'
    ):
    """
    用于打印下载进度条的函数
    @param iteration: 当前的迭代次数
    @param total: 总共的迭代次数
    @param prefix: 进度条前缀
    @param suffix: 进度条后缀
    @param decimals: 进度数值的小数位数
    @param length: 进度条的长度
    @param fill: 进度条的填充字符
    @param print_end: 进度条结束时的字符
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total:
        print('\n')

def test():
    # 模拟下载10个文件
    total_files = 10
    for i in range(total_files):
        download_file()
        print_progress_bar(
            i + 1, 
            total_files, 
            prefix='Progress:', 
            suffix='Complete', 
            decimals=2,
            length=50,
            fill = "█"
        )

if __name__ == '__main__':
    test()