import time
import torch
from network_NN import CNN
from torch.utils.data import DataLoader
from torchvision import transforms ,datasets

######################### CNN数字分类 #############################
#一般步骤：1.加载数据  2.初始化网络 3.设置损失函数 4.设置优化器 5.开始训练（载入数据；梯度清零；数据前向传播；计算损失；反向传播；参数优化）
if __name__ =="__main__":
    #超参数设定
    num_epoch = 5   #所有样本的迭代次数
    batch_size = 64 #批量的大小即一次性输入到网络中的样本数
    output_channel = 10 #全连接层的输出
    l2_reg = 0.01  # L2 正则化系数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #加载数据
    transform = transforms.Compose([transforms.ToTensor() , transforms.Normalize((0.5,) , (0.5,))]) #数据预处理工具，它可以将多个数据预处理操作组合起来
    data_train = datasets.MNIST(root='data/' ,transform=transform , train=True , download=True) #下载数据
    data_test = datasets.MNIST(root='data/' ,transform=transform ,train=False ,download=True)
    data_train_load = DataLoader(dataset=data_train , batch_size=batch_size , shuffle=True) #加载数据
    data_test_load = DataLoader(dataset=data_test ,batch_size=batch_size , shuffle=True)

    #初始化神经网络
    Net = CNN( output_channel ) #初始化网络
    Net = Net.to(device)    #模型放至GPU

    #设置损失函数
    loss_fn = torch.nn.CrossEntropyLoss()   #交叉熵

    #设置优化器
    optimizer = torch.optim.Adam(Net.parameters())  #Adam优化器

    all_time_start = time.time()
    #开始训练
    print("开始训练")
    for epoch in range(num_epoch):
        cost = 0
        time_start = time.time()
        for data in data_train_load:
            input , labels = data
            input= input.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()   #梯度清零
            out = Net(input)        #前向传播
            loss = loss_fn(out , labels)   #计算损失

            # 添加 L2 正则化项
            l2_loss = 0
            for param in Net.parameters():  #
                l2_loss += torch.norm(param, 2) #torch.norm求范数，2代表幂的次数
            loss = loss + l2_reg * l2_loss

            loss.backward() #反向传播
            optimizer.step()    #参数优化
            cost = cost + loss.data
        time_end =time.time()
        print(f"epoch:{epoch+1} loss:{'%.2f'%(cost / len(data_train_load))} timespent:{'%.2f'%(time_end - time_start)}s")

    all_time_end =time.time()
    print(f"训练结束 traintime:{'%.2f'%(all_time_end -all_time_start)}s")

    print("开始测试")
    correct = 0
    total =0
    true_positive = 0
    false_negative = 0
    Net.eval()
    with torch.no_grad():
        for data in data_test_load:
            input, labels = data  # 载入数据
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
