## 这是cifar-10图像分类竞赛
### 来源于kaggle
> todolist 22/8
- [x] 搭建项目框架
- [x] 从kaggle下载数据

>todolist 25/8
- 搭建模型 如下几种方式：
- 1. 从model.yaml文件直接加载cfg配置文件
- 2. resnet18.py文件解析yaml文件,转换为Python中的字典或列表结构，
- 3. 利用models.common定义基本模块
- 4. 利用parse_model函数解析yaml文件中的配置，创建网络层并组合
- 5. 初始化模型参数

> todolist 26/8
### 对于cifar10项目
1. [x] 先从torchversion.models加载resnet18模型，再改变shape(nc=10)
2. [x] yaml文件构建模型，再做裁剪，构造模型

> todolist 28/8
#### 早停机制：通常监控验证集的损失（或准确率等度量指标）
1. [x] 完成early stopping 功能
2. [x] 完成cutoff 功能
1. 耐心（Patience）
2. 最小改善量（Minimum Delta)


