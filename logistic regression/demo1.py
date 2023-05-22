import numpy as np

#############此版本为手撕逻辑回归############

def load_data(data_pth):
    data = np.loadtxt(data_pth , delimiter=',') #下载数据，delimiter起到分割数据作用
    x = data[ : , :2]           #取所有行和前两列作为x
    y = data[ : , 2]            #取所有行和第三列为y
    return x,y

def z_score_nomalization(x):
    u = np.mean(x , axis=0)  #对每一个维度的输入特征取平均，即每一列，结果为1x2
    std = np.std(x , axis=0)    #对每一个维度的输入特征取标准差，即每一列，结果为1x2
    x_norm = (x - u) / std    #对每一行的数据按照此公式进行计算
    return x_norm

def gradient_decent(x_norm , y , num_iter , lr , lambda_):
    m,n = x_norm.shape #m为行数，n为列数
    w = np.zeros(n)     #设置初始的w值
    b = np.zeros(1)     #设置初始的b值
    for i in range(num_iter):    #开始迭代训练
        dj_w , dj_b , cost= gradient_function(x_norm , y , w ,b ,m ,n ,lambda_) #求出梯度值和损失值
        w = w - lr * dj_w   #更新参数
        b = b - lr * dj_b
        print(f'Iteration{i+1} cost :{abs(cost)}')
    return w , b

def gradient_function(x_norm , y , w ,b ,m ,n ,lambda_):
    erro = 0
    dj_w = np.zeros(n)
    dj_b = 0
    for i in range(m):
        erro =  sigmoid(np.dot(x_norm[i] , w) + b)  - y[i]  #求出损失值，课程中这里用的仍然是平方误差，可能是由于逻辑回归代价过于复杂，这里仍然使用了简单的计算方法
        for j in range(n):
            dj_w[j] = dj_w[j] + erro * x_norm[i,j]    #代价函数对不同w求偏导时会产生不同的乘积，所以要分开计算
        dj_b = dj_b + erro
    dj_w =dj_w / m  + ( lambda_ / m ) * w       #加上正则项保证不同维度特征的“等价”影响
    dj_b = dj_b / m
    return dj_w ,dj_b ,erro

def sigmoid(z):
    y = 1.0 / (1.0 + np.exp(-z))
    return y

if __name__ == "__main__":

    #数据地址与超参数设定
    tran_data = 'data/ex2data1.txt' #数据地址
    test_data = 'data/data1_test.txt'
    num_iter = 200      #超参数设定，训练轮数
    lr = 0.1         #学习率
    lambda_ = 0.5       #正则项参数

    #训练模型
    x , y = load_data(tran_data)    #加载训练数据
    x_norm =z_score_nomalization(x) #输入特征归一化，加快与稳定训练
    w,b = gradient_decent(x_norm , y , num_iter , lr , lambda_) #求取目标参数

    #预测
    x_test , y_test = load_data(test_data)  #加载测试数据
    x_test_norm = z_score_nomalization(x_test)  #对测试数据进行特征归一
    m , n = x_test_norm.shape
    y_predict = np.zeros(m) #初始化预测数组
    for i in range(m):
        y_predict[i] = sigmoid( np.dot(w , x_test_norm[i]) + b)
        if y_predict[i] >=0.5:  #决策边界
            y_predict[i] = 1
        else:
            y_predict[i] = 0
    print(f"predict:{y_predict}",f"true:{y_test}")  #结果精度较差