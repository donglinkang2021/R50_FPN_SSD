# README

## 项目简单介绍

![Resnet50_FPN_SSD](README.assets/Resnet50_FPN_SSD.png)

这是北京理工大学深度学习基础课程最后一次大作业目标检测比赛，详细介绍在[这里](./比赛要求)，这里自己主要用到了华为云的Modelarts和colab的算力来训练自己的数据，将自己这段时间做的工作开源出来：

- 从零开始实现了Resnet50+FPN+SSD的目标识别模型，具体介绍可以看`大作业-董林康-1120212477.ipynb`
- 用gradio做了一些可视化锚框的界面设计

## 文件目录

```shell
PS D:\Desktop2\DL_Foundation\assignment\lastwork> ls


    目录: D:\Desktop2\DL_Foundation\assignment\lastwork


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----         2023/7/15     12:30                checkpoint
d-----         2023/7/14     17:31                data
d-----         2023/6/20     15:27                flagged
d-----         2023/7/15     12:34                huaweilab
d-----          2023/7/7      9:49                image
d-----         2023/6/27     18:39                learning
d-----          2023/7/4     12:37                mydet
d-----         2023/7/14     16:02                README.assets
d-----         2023/6/30     20:43                reference
d-----         2023/7/15     12:34                submit
d-----          2023/7/6     16:42                utils
d-----         2023/6/20     18:28                __pycache__
d-----         2023/7/14     17:22                数据和提交内容
-a----         2023/7/15     12:45             83 .gitignore
-a----          2023/7/6     21:52         265120 0查看数据集.ipynb
-a----          2023/7/6     15:50        2768763 1查看提交格式.ipynb
-a----         2023/6/29     21:03        1038783 2_SSD_colab_demo.ipynb
-a----          2023/7/4      8:52         129673 3_1_choose_backbone.ipynb
-a----          2023/7/4     12:19         356725 3_2_mydet_demo.ipynb
-a----          2023/7/5      0:21         202731 3_3_mydet_cpu.ipynb
-a----          2023/7/5      0:40         610098 3_4_mydet_colab.ipynb
-a----          2023/7/6     22:16        1490960 3_5_mydet_eval.ipynb
-a----          2023/7/6     16:24          25088 3_6_mydet_eval_batch.ipynb
-a----          2023/7/6     18:48          29565 3_7_mydet_eval_batch_submit.ipynb
-a----          2023/7/7      9:49           3157 anchor_size_ratios.ipynb
-a----          2023/7/1      0:22           1807 DailyNote.md
-a----         2023/7/15     12:32           3996 README.md
-a----         2023/6/26     21:39             30 requirements.txt
-a----         2023/7/15     13:36       55750793 starting_kit.zip
-a----          2023/7/6     16:19         441808 test_images.csv
-a----          2023/7/7     11:32         798937 大作业-董林康-1120212477.ipynb
-a----         2023/6/18     17:14           2605 比赛要求.md

```

### main

- `大作业-董林康-1120212477.ipynb`为本次提交的最终作业，其中的图片在`image`文件夹下，建议解压附件后阅读。
- `mydet`和`utils`是最后所有提交文件都用到的包，包括乐学要求提交的`大作业-董林康-1120212477.ipynb`里面的代码也调用了其中的代码，包中代码均为手写，小部分工具函数参考了李沐的d2l的实现过程。
- `image`文件夹中是自己用draw.io或者plt画的一些图

### others

- `huaweilab`文件夹为自己在华为云的上配置的环境目录（除了数据集）
- `reference`是在写代码中参考的三篇论文，分别是Cascade RCNN原论文，Resnet原论文，SSD原论文
- `submit`文件夹中包括了自己最终没有提交成功的两个结果，包括colab和modelarts的结果
- 其余的`learning`文件夹中的`ipynb`和上面目录中的其它`ipynb`文件均为一点一点做该项目的学习notebook，包含具体文件代码及思路注释
- `requirements.txt`为运行本项目所需要的python包
- `test_images.csv`为自己生成的测试集图片信息的csv文件
- `checkpoint`文件夹中是colab和modelarts训练的参数，由于超过一百兆，没有上传github
- `starting_kit.zip`中是助教一开始给的读取标注的样例文件
