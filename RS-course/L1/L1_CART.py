
# -*- coding: utf-8 -*-
# 使用CART进行MNIST鸢尾花分类
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
data = digits.data
print(data.shape)
# 查看第一幅图像
print(digits.images[0])
# 第一幅图像代表的数字含义
print(digits.target[0])
# 将第一幅图像显示出来
plt.gray()
plt.imshow(digits.images[0])
plt.show()

# 分割数据，将25%的数据作为测试集，其余作为训练集
X_train, X_test, y_train, y_test = train_test_split(
digits.data, digits.target, test_size=0.25, random_state=33)

# 采用Z-Score规范化
# ss = preprocessing.StandardScaler()
# train_ss_x = ss.fit_transform(X_train)
# test_ss_x = ss.transform(X_test)

# 创建决策树
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

#模型预测
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

