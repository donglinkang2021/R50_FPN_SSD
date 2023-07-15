# README

华为云ModelArts使用的一些心得：

## 上传数据

非常方便，相比于colab，华为云的数据只需要上传到OBS中，然后新建笔记本的时候再里面上传的时候可以调用OBS的数据。

## 每次退出后再进入

原来的数据还在，不会马上删除，这一点做的也比colab好，毕竟都是花了钱的（代金券），但之前配置过的环境比如pip install过的包会消失，所以这里自己要重新在terminal中配置环境：

```shell
pip install -r requirements.txt
pip uninstall numpy -y
pip install numpy
```

## jupyterlab

华为云的modelarts可以上传并运行多个notebook

## Code

最喜欢的特点就是可以利用Vscode进行远程开发，只需要配置一下ssh即可，网上都有对应教程。而且在线上的lab环境也做的不错，和Vscode一样都有不错的代码注释提示效果。

## 一些奇怪的报错

这里的pytorch版本固定是1.8，所以会发生一些奇怪的错误，和本地和colab某些用法不太一样，比如在华为云上才报错的：


```shell
~/work/mydet/anchor.py in generate_anchors_prior(self, fmap_h, fmap_w, device)
     59         center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
     60         center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
---> 61         shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
     62         shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
     63 

TypeError: meshgrid() got an unexpected keyword argument 'indexing'
```

方法是在对应的包中把`indexing='ij'`删掉即可。