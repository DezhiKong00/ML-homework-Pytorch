import numpy as np
from sklearn.linear_model import LogisticRegression

def load_data(data_pth):
    data = np.loadtxt(data_pth , delimiter=',') #下载数据，delimiter起到分割数据作用
    x = data[ : , :2]           #取所有行和前两列作为x
    y = data[ : , 2]            #取所有行和第三列为y
    return x,y

if __name__ =="__main__":
    tran_data = 'data/ex2data1.txt' #训练数据地址
    test_data = 'data/data1_test.txt' #测试数据地址
    x , y = load_data(test_data)    #记载训练数据
    x_test , y_test = load_data(test_data)  #加载测试数据
    lr_model = LogisticRegression() #记载模型
    lr_model.fit(x , y) #训练数据
    y_predict = lr_model.predict(x_test) #逻辑回归进行预测
    print(f"predict:{y_predict}" , f"true:{y_test}")    #精度较高，可能与超参数的设定以及使用逻辑损失有关
