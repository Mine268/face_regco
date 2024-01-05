# README

高级人工智能课程 第十四组 大作业 多人种人脸识别

任凯文 曲涵石 罗凡 马彩莲

## 数据准备

`split.tar.gz` 数据下载

```
https://1drv.ms/u/s!Aom6BpCyKWQzgiffwPdYN3fjJYgw?e=bImCiI
```

解压在项目根目录下。

## 环境搭建

使用 conda 建立环境

```
conda create -n common python=3.10
```

在根目录下运行环境配置

```bash
conda activate common
pip install -r requirements.txt
```

## 训练

创建文件夹存放模型

```bash
mkdir checkpoints
```

在项目根目录下运行

```bash
python train_vit.py --data_path train_split --batch_size 64 --epoches 20 --lr 0.0001
```

## 测试

在根目录下运行

```bash
python test.py
```

会输出前 20 轮模型对应的 AUC 指标和第 20 轮模型的 ROC 曲线（TAR + FAR）

# 分工

模型 & 代码：任凯文、罗凡

实验 & 文档：曲涵石、马彩莲