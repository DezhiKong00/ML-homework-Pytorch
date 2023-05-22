from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载红酒质量数据集
wine = load_wine()
print(wine)
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)

# 构建回归树模型
regressor = DecisionTreeRegressor(random_state=42)

# 训练回归树模型
regressor.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = regressor.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print('预测值',y_pred)
print('均方误差：', mse)