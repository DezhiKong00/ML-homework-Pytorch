import numpy as np
#############此版本为手撕线性回归############
def load_data(data_pth):
    data = np.loadtxt(data_pth , delimiter= ',')    #下载数据，delimiter起到分割数据的作用
    x = data[ : , 0:4]  #数据前四列的输入特征，即'size(sqft)','bedrooms','floors','age'。
    y = data[ : , 4]    #数据第五列为输出特征，即'cost'
    return x , y

def z_score_normalization(x):
    u = np.mean(x , axis=0)     #对数据的每一列求均值（即对相同特征求均值,axis=0代表对每一列求均值，axis=1代表对每一行求均值），结果压缩为一行
    std = np.std(x , axis=0)    #对数据的每一列求标准差，结果压缩为一行
    x_norm = (x - u) / std      #numpy每一行的数据都会按照此公式进行计算
    # print(x_norm)             #根据需要可以进行可视化查看
    return x_norm

def gradient_descent(x_norm , y , num_iter , lr , lambda_):
    m , n = x_norm.shape    #求出x_norm的矩阵行列，其中m为行，n为列
    w = np.zeros(n)         #初始化参数为0，w与输入特征相同维度
    b = 0
    for i in range(num_iter):   #进行迭代训练
        dj_w , dj_b , cost= gradient_function(x_norm , y , m , n , w , b , lambda_)     #由梯度函数求取梯度
        print(f'Iteration{i+1} cost :{abs(cost)}')  #打印每一次的损失值
        w = w - lr * dj_w       #参数更新,w为向量包含四个参数
        b = b - lr * dj_b
    return w , b


def gradient_function(x_norm , y , m , n , w , b , lambda_):
    dj_w = np.zeros(n)  #跟w相同的维度
    dj_b = 0
    for i in range(m):  #先对每一行求取w1,w2,w3,w4 之后在求和取平均
        cost = np.dot(x_norm[i] , w) + b - y[i]     #求出代价值
        for j in range(n):
            dj_w[j] = dj_w[j] + cost * x_norm[i , j]    #cost对不同w求偏导时会产生不同的乘积，所以要分开计算
        dj_b = dj_b + cost
    dj_w = dj_w / m
    dj_b = dj_b / m

    for t in range(n):          #添加正则项
        dj_w = dj_w + (lambda_/m) * w[t]
    return dj_w ,dj_b ,cost

if __name__ == '__main__':

    #数据地址与超参数
    train_pth = 'data/houses.txt'  # 训练数据的路径
    test_pth = 'data/test.txt'
    num_iter = 200      #迭代次数，超参数可以修改，这里的参数并不是最好的
    lr = 0.0175       #学习率
    lambda_ = 0.7     #正则项参数

    #训练模型
    x,y = load_data(train_pth)   #加载训练数据
    x_norm = z_score_normalization(x)    #将输入特征归一化以加快和稳定训练，这里使用z-score，可以根据需要进行更改
    w , b = gradient_descent(x_norm , y , num_iter , lr ,lambda_) #求取目标参数w与b（注意此处w为向量w=（w1，w2，w3，w4））,loss记录每一次的损失值

    #预测
    x_test , y_test = load_data(test_pth)    #加载测试数据
    x_test_norm = z_score_normalization(x_test)     #对测试数据归一化
    y_predicate = np.dot(x_test_norm , w) + b         #预测
    print(f"predict:{y_predicate}",f"true:{y_test}")
