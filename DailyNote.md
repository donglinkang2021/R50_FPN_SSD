# 开发日志

## 6.26 

这里记录一下使用colab的一些小心得：

- 上传数据集到google drive的时候上传一个zip要比上传文件夹快得多（理所应当的经验是自己用这轻薄本上传了一个晚上才发现的）

- 这里打包的文件目录有（colab的notebook只能有一个，而自己手写的包可以用多次，所以要上传一点本地的东西）

  ```shell
  image
  utils_d2l
  requirements
  ```

- colab分割cell的技巧，先按<kbd>ctrl</kbd>+<kbd>M</kbd>再按<kbd>-</kbd>

## 6.28

上传压缩包到google drive之后，在colab里面解压的话不需要每次先从drive的文件夹中复制过来，直接解压到一个目录下就可以了，这样就不会占用colab的空间了。比如：

```python
# 将./drive/MyDrive/dl_detection.zip先复制到当前目录下再解压
filename = './drive/MyDrive/dl_detection.zip'

# 复制文件 花了 2 min 27s
import shutil
shutil.copyfile(filename, './dl_detection.zip')

# 解压缩文件 大概 1 min 21s
with zipfile.ZipFile('./dl_detection.zip', 'r') as zip_ref:
    zip_ref.extractall('./')
```

下面的代码更快：

```python
# 将./drive/MyDrive/dl_detection.zip解压到当前目录下
filename = './drive/MyDrive/dl_detection.zip'
# 解压缩文件 大概 2 min 这个更快
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall('./')

```

## 6.30

理解`.contiguous()`的作用，起因是自己不理解head中写的代码，这里记录一下：

```python
cls_scores.append(cls_layer(feature).permute(0, 2, 3, 1).contiguous())
bbox_preds.append(bbox_layer(feature).permute(0, 2, 3, 1).contiguous())
```

[参考这里](https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch)