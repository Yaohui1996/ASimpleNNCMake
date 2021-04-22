# 一个神经网络的实现(C++)
## 写在前面
- 可以使用该程序构建**任意层数**、**任意神经元数目**的神经网络(只要内存足够！)
- 理论上讲，只要输入采用one-hot encoding的方式，任务目标为有监督的多分类，该网络都可行
## 环境和依赖
- CMake==3.13.4
- GCC==8.3.0
- Eigen==3.3.8

程序在上述环境下可编译通过并正确运行。


## 例子
以MNIST数据集为例：
- 5层网络
- 神经元数目分别为{784, 64, 128, 64, 10}
- (-1,1)均匀分布初始化w和b
- 步长eta取0.1
- 步长衰减系数alpha取0.9
- 训练30次

我运行的那一次：

训练集准确率：0.991367

测试集准确率：0.9682


## 参考
1. 李航《统计学习方法》(第二版)
2. Michael Nielsen《Neural Networks and Deep Learning》http://neuralnetworksanddeeplearning.com/index.html
