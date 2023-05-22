import torch.nn


class MLP(torch.nn.Module):
    def __init__(self , input_layer , out_layer ):
        super(MLP, self).__init__()
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_layer ,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100 ,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100 ,out_features=out_layer),
            # torch.nn.Softmax()
        )

    def forward(self , x):
        out = self.MLP(x)
        return out

class CNN(torch.nn.Module):
    def __init__(self,out_channel):
        super(CNN, self).__init__()
        self.CNN = torch.nn.Sequential(
            torch.nn.Conv2d(1 ,32 ,5 , 1 ), #24*24
# 在神经网络中，卷积和池化操作的输入是一个四维张量，其维度依次为(batch_size, in_channels, height, width)，其中batch_size
# 表示当前输入的数据批次大小，in_channels表示输入数据的通道数，height和width分别表示输入数据的高度和宽度。
# 在这个模型中，卷积和池化操作的输入张量的大小为(batch_size, 1, 28, 28)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  #12*12
            torch.nn.Conv2d(32 , 64 ,5 , 1), #8*8
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  #4*4
            torch.nn.Flatten(),
            torch.nn.Linear(64*4*4 , 64 ),
            #在这个模型中，卷积层的输出的大小为 (batch_size, 64, 4, 4)，其中 batch_size
            #表示当前输入的数据批次大小，64 表示卷积层输出的通道数，4 表示卷积层输出的高度和宽度。
            #torch.nn.Linear(64*4*4 , 64 )这种写法能够自动计算输入大小，而其他的卷积和池化也是
            torch.nn.Linear(64 , out_channel)
        )

    def forward(self , x):
        out = self.CNN(x)
        return out