import numpy as np
from sklearn.linear_model import LinearRegression

#############使用sklearn库非常easy#############
def load_data(data_pth):
    data = np.loadtxt(data_pth , delimiter= ',')    #下载数据，delimiter起到分割数据的作用
    x = data[ : , :4]  #数据前四列的输入特征，即'size(sqft)','bedrooms','floors','age'。
    y = data[ : , 4]    #数据第五列为输出特征，即'cost'
    return x , y


if __name__ == "__main__":

    #数据地址与超参数
    train_pth = 'data/houses.txt'   #训练数据路径
    test_pth =  'data/test.txt'     #测试数据路径

    #训练模型
    x,y = load_data(train_pth)      #加载训练数据
    linear_model = LinearRegression()   #加载模型，参数可以修改，这里直接使用默认参数
    linear_model.fit(x,y)           #训练模型

    #测试模型
    x_test,y_true = load_data(test_pth)     #加载测试数据
    y_test = linear_model.predict(x_test)   #对测试数据进行预测
    print(f"predict:{y_test}",f"true:{y_true}")
