## 这是cifar-10图像分类竞赛
### 来源于kaggle
> todolist 22/8
- [x] 搭建项目框架
- [x] 从kaggle下载数据

>totolist 25/8
- 搭建模型 如下几种方式：
- 一，从model.yaml文件直接加载cfg配置文件
- 二，加载预训练模型再改变输出的shape(nc: number of class)
- 三，利用models.common模块构造classify(nn.Module)函数

### 对于cifar10项目
1. 先搭建resnet18组件，再构造分类函数，再改变shape(10)
2. [x] 先从torchversion.models加载resnet18模型，再改变shape(nc=10)
3. 先构造yaml配置文件，加载配置文件构造模型，再构造分类函数，再改变shape(10)
