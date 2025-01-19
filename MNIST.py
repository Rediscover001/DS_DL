# 导入必要的库
from datetime import datetime

import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 定义超参数
batch_size = 1  # 每批次的样本数量
learning_rate = 0.01  # 学习率
num_epochs = 2  # 训练轮数

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化处理
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)



# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # 全连接层，输入28x28，输出512
        self.fc2 = nn.Linear(512, 1024)  # 全连接层，输入512，输出256
        self.fc3 = nn.Linear(1024, 10)  # 全连接层，输入256，输出10（
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将图像展平为一维向量
        x = F.relu(self.fc1(x))  # 第一层全连接 + ReLU激活
        x = self.fc2(x)  # 第二层全连接
        x = self.fc3(x)  # 第三层全连接（输出层）

        return x

# 初始化模型、损失函数和优化器
model = Net()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降优化器

# 训练模型
for batch_size in range(16,17):
    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    t1 = time.time()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 前向传播
            model.train()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            # 每100个批次打印一次损失
            """
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            """
    # 测试模型
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果,dim = 1即类别维度 num_calsses;if dim = 0,batch_size维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        #print(f'Test Accuracy: {100 * correct / total:.2f}%')
        Test_Accuracy=100 * correct / total
    t2 = time.time()
    print(f"Time taken for {num_epochs} epoch {t2 - t1}")

    with open(file = "log.txt", mode = "a+") as log:
        log.write(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Batch:{batch_size} | Epochs:{num_epochs} | LR:{learning_rate} | Running Time: {(t2 - t1):.2f}s | Accuracy:{Test_Accuracy}\n")
"""
代码说明

    数据预处理:

        transforms.ToTensor(): 将图像转换为PyTorch张量。

        transforms.Normalize(): 对图像进行标准化处理。

    模型定义:

        Net类继承自nn.Module，定义了三个全连接层。

        forward方法定义了数据的前向传播过程。

    训练过程:

        使用交叉熵损失函数和随机梯度下降优化器。

        每个epoch中遍历训练数据，计算损失并更新模型参数。

    测试过程:

        在测试集上评估模型性能，计算准确率。

运行结果

    训练过程中会打印每个epoch的损失。

    测试结束后会打印模型在测试集上的准确率。

注意事项

    确保已安装PyTorch和TorchVision库。

    可以根据需要调整超参数（如batch_size、learning_rate等）以优化模型性能。
"""

"""
Q1:why normalizing the pictures is needed

"""
"""
Q2:How many loss functions there are,and how to chose one of them to use?
如何选择合适的损失函数
（1）根据任务类型选择
    回归任务：MSE、MAE、Huber Loss。
    分类任务：交叉熵损失、二元交叉熵损失、NLLLoss。
    序列任务：CTC 损失、交叉熵损失。
    生成任务：对抗损失、L1/L2 损失。

（2）根据数据分布选择
    如果数据中存在异常值，选择对异常值不敏感的损失函数（如 MAE、Huber Loss）。
    如果数据分布不平衡，可以使用加权损失函数（如加权交叉熵损失）。

（3）根据模型输出选择
    如果模型输出是概率分布（如 Softmax 输出），使用交叉熵损失。
    如果模型输出是连续值（如线性回归），使用 MSE 或 MAE。
    
（4）根据训练目标选择
    如果目标是最大化分类准确率，使用交叉熵损失。
    如果目标是生成逼真的图像，使用对抗损失。
"""

"""
Q2:How many loss functions there are,and how to chose one of them to use?
如何选择合适的优化器
（1）根据任务复杂度选择
    简单任务：SGD 或带动量的 SGD。
    中等复杂度任务：Adam、RMSProp。
    复杂任务：Adam、AdamW。
    
（2）根据数据特性选择
    稀疏数据：Adagrad、Adam。
    非平稳目标函数：RMSProp、Adam。
    
（3）根据模型特性选择
    需要正则化：AdamW。
    需要快速收敛：Adam、RMSProp。
    
（4）根据实验效果选择
    在实际应用中，可以尝试多种优化器，选择效果最好的一个。

"""

"""
Q3:why I need creat dataloaders
1. 高效的数据加载
    问题：直接加载整个数据集可能会占用大量内存，尤其是当数据集非常大时（如图像、视频、文本数据集）。
    解决方案：DataLoader 可以按需加载数据（即每次只加载一个批次的数据），从而减少内存占用。

2. 批处理（Batching）
    问题：深度学习模型通常需要以批次（batch）的形式输入数据，而不是单条数据。
    解决方案：DataLoader 可以将数据集划分为多个批次，方便模型训练。
        例如，如果数据集有 1000 个样本，batch_size=100，则 DataLoader 会将其划分为 10 个批次。

3. 数据打乱（Shuffling）
    问题：如果数据是按某种顺序排列的（如按类别排序），模型可能会学习到这种顺序信息，而不是真正的数据分布。
    解决方案：DataLoader 可以通过设置 shuffle=True 在每个 epoch 开始时打乱数据顺序，从而避免模型对数据顺序产生依赖。

4. 并行加载数据
    问题：数据加载可能成为训练过程的瓶颈，尤其是当数据预处理较复杂时。
    解决方案：DataLoader 支持多线程数据加载（通过 num_workers 参数），可以显著加快数据加载速度。

5. 数据预处理
    问题：原始数据通常需要经过预处理（如归一化、数据增强）才能输入模型。
    解决方案：DataLoader 可以与 Dataset 结合使用，在数据加载时自动应用预处理操作。

6. 灵活的数据集管理
    问题：数据集可能分布在多个文件或目录中，手动管理这些数据非常麻烦。
    解决方案：DataLoader 可以方便地管理数据集，支持从文件、目录、内存等多种数据源加载数据。
"""