from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits    # 注意加载数据的时候应该从数据集里面加载
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# 导入预处理工具，将数据进行规范化
from sklearn import preprocessing  
# 导入评价准确率的指标
from sklearn.metrics import accuracy_score

# 加载数据
digits = load_digits()
data = digits.data
# 数据探索
print(data.shape)
# 查看第一幅图像
print(digits.images[0])
# 第一幅图像代表的数字含义:代表数字0
print(digits.target[0])
# 将第一幅图像显示出来
plt.gray()
plt.imshow(digits.images[0])
plt.show()
# 分割数据，将25%的数据作为测试集，其余作为训练集（当然也可以指定其他比例。）
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)
# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
# 使用训练集进行拟合
train_ss_x = ss.fit_transform(train_x)
# 使用测试集进行验证
test_ss_x = ss.transform(test_x)

# 创建KNN分类器,参数可以使用默认，也可以自己制定，大部分情况下使用默认即可（括号中保持为空）
# 如果将K值甚至成为200，分类的准确度就会降低很多！KNN 准确率：0.8489
knn = KNeighborsClassifier(n_neighbors=5, weights= 'uniform', algorithm='auto',leaf_size=30)
knn.fit(train_ss_x, train_y)
predict_y = knn.predict(test_ss_x)
print("KNN 准确率：%.4lf" % accuracy_score(test_y, predict_y))

# 创建SVM分类器（高斯核函数,其系数为gamma='auto'）
svm = SVC(kernel='rbf',C=1.0,gamma='auto')
svm.fit(train_ss_x, train_y)
predict_y = svm.predict(test_ss_x)
print('SVM 准确率：%0.4lf' % accuracy_score(test_y, predict_y))

# 采用Min—Max规范化，因为多项式贝叶斯不能传入负数
mm = preprocessing.MinMaxScaler()
train_mm_x = mm.fit_transform(train_x)
test_mm_x = mm.transform(test_x)

# 创建 Navie Bayes分类器
mnb = MultinomialNB()
mnb.fit(train_mm_x, train_y)
predict_y = mnb.predict(test_mm_x)
print("多项式朴素贝叶斯准确率：%.4lf" % accuracy_score(test_y,predict_y))

# 创建CART决策树分类器
dtc = DecisionTreeClassifier()
dtc.fit(train_mm_x, train_y)
predict_y = dtc.predict(test_mm_x)
print("CART 决策树的准确率为：%.4lf" % accuracy_score(test_y,predict_y))
