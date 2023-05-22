import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 读取Excel文件并将数据存储到DataFrame中
data = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 将数据集中的类别变量进行编码
label_encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = label_encoder.fit_transform(data[col])
print(data)

# 将编码后的数据集中的类别变量进行独热编码
onehot_encoder = OneHotEncoder(sparse=False)
X = data.drop('love', axis=1)
X = onehot_encoder.fit_transform(X)
y = data['love']

# 构建决策树模型
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X, y)

# 使用模型对新数据进行预测
# 假设新数据为
new_data = [[1,2,0,1] , [0,0,1,0]]
new_data = onehot_encoder.transform(new_data)
prediction = clf.predict(new_data)
print("预测结果为：", prediction)