# 导入必要的库
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# 生成模拟数据集
x,y = make_classification(n_samples=1000 , n_features=6 ,n_informative=2 , n_redundant=0,random_state=42 ,shuffle=False)

# 将数据集分成训练集和测试集
x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size=0.2 , random_state=42)

# 创建GBDT模型
gbdt =GradientBoostingClassifier(n_estimators=100 , learning_rate=0.1 , max_depth=3 ,random_state=42)

# 训练模型
gbdt.fit(x_train, y_train)

# 预测测试集
y_pred = gbdt.predict(x_test)

# 评估模型性能
score = gbdt.score(x_test, y_test)
print("Accuracy:", score)