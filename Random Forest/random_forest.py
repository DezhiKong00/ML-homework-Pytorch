# 导入必要的库
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成模拟数据集
X, y = make_classification(n_samples=1000, n_features=6,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 创建随机森林模型
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0)

# 训练模型
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 评估模型性能
score = rf.score(X_test, y_test)
print("Accuracy:", score)