import time
import numpy as np
import torch
from network_NN import MLP
from torch.utils.data import  DataLoader
from torchvision import transforms, datasets

########################### MLP数字分类 ############################
#一般步骤：1.加载数据  2.初始化网络 3.设置损失函数 4.设置优化器 5.开始训练（载入数据；梯度清零；数据前向传播；计算损失；反向传播；参数优化）
if __name__ == "__main__":
    # 超参数设定
    num_epoch = 5           #总样本训练轮数
    batch_size = 128       #一次性输入到网络中的样本数
    input_layer = 28 * 28   #输入层神经元个数
    out_layer = 10          #输出层神经元个数
    l2_reg = 0.01           #正则化系数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 判断是否有GPU可用

    # 加载数据
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])   #数据预处理工具，它可以将多个数据预处理操作组合起来
    data_train = datasets.MNIST(root="data/", transform=transform, train=True, download=True)       #下载数据
    data_test = datasets.MNIST(root="data/", transform=transform, train=False, download=True)
    data_train_load = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)   #加载数据
    data_test_load = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

    # 初始化神经网络
    Net = MLP( input_layer ,out_layer)  #初始化网络
    Net = Net.to(device)  # 将网络放入GPU运算

    # 设置损失函数
    loss_fn = torch.nn.CrossEntropyLoss()   #交叉熵

    # 设定优化器
    optimizer = torch.optim.Adam(Net.parameters())  #亚当优化器，其中可以设置初始的学习速率，此处未设置

    all_star_time = time.time() #计算训练时间

    print("开始训练")
    # 开始训练
    for epoch in range(num_epoch):
        cost = 0    #计算每一轮的代价
        start_time = time.time()  # 计算每一轮的训练时间
        for data in data_train_load:
            input , labels = data   #加载数据,数据大小为 torch.Size([128, 1, 28, 28])
            input = input.to(device)    #将数据放在GPU上进行训练
            input = torch.flatten(input , start_dim=1)  #展平数据，数据大小为 torch.Size([128, 784])

            labels = labels.to(device)  #将标签也放入GPU上训练

            optimizer.zero_grad()   #梯度清零
            y_predict = Net.forward(input)  #数据前向传播

            loss = loss_fn(y_predict, labels)   #计算损失值

            #添加 l2 正则化项
            l2_loss = 0
            for param in Net.parameters():
                l2_loss += torch.norm( param , 2)
            loss += l2_reg*l2_loss

            loss.backward()     #反向传播求出梯度
            optimizer.step()    #优化器对参数进行更新
            cost = cost + loss.data     #计算总的代价值，总的loss个数为 len（data_train） / 128
            end_time =time.time()
        print(f"epoch:{epoch + 1} loss:{'%.2f' %(cost/len(data_train_load))} timespent:{'%.2f' %(end_time-start_time)}s")
    all_end_time = time.time()
    print(f"训练结束 total time spent:{'%.2f' % (all_end_time - all_star_time)}s")

    print("开始测试")
    #评测模型
    correct = 0
    total =0
    true_positive = 0
    false_negative = 0
    Net.eval()  #评测
    with torch.no_grad():
        for data in data_test_load:
            input, labels = data  # 载入数据
            input = torch.flatten(input, start_dim=1)  # 展平数据
            input = input.to(device)  # 将数据放在GPU上进行测试
            labels = labels.to(device)

            out = Net.forward(input)  # 前向传播
            _, id = torch.max(out.data, 1)  # 计算出每一行的最大值即预测的数字。其对应的序号为数字
            correct += torch.sum(id == labels.data)  # torch.sum可以计算出 id == labels.data的样本数量
            total += len(labels)
            true_positive += torch.sum((id == labels.data) & (id == 1))
            false_negative += torch.sum((id != labels.data) & (id == 0))

    accuracy = 100 * correct / total
    recall = 100 * true_positive / (true_positive + false_negative)

    print(f"测试结束 accuracy: {'%.2f' % accuracy}% recall: {'%.2f' % recall}%")

