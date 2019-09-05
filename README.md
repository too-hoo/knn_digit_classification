# 使用KNN：对手写数字进行识别

## 在sklearn中使用KNN
sklearn使用
- 分类:KNeighborsClassifier,回归：KNeighborRegressor
- KNeighborClassifier构造参数：
    - n_neighbors:即KNN中的K值，默认是5
    - weights：用来确定邻居的权重，三种方式：'uniform','distance',自定义函数
    - algorithm：规定计算邻居的方法，四种方式:'auto','kd_tree','ball_tree','brute'
    - leaf_size:代表构造KD树或者球树的叶子数，默认是30
- KNeighborsClassifier功能函数：
    - fit(train_x, train_y):分类器的训练
    - predict(test_x):用训练好的分类器进行预测

## 用KNN对手写数字进行识别分类
数据集
- 手写数字数据集，将数字图像0-9进行对应
- 采用sklearn自带的简版数据集（digits），1797幅图，每幅图8*8的矩阵

项目流程
- 准备阶段：数据探索，数据可视化、数据规范化
- 分类阶段：特征选择，模型训练，结果评估

## 总结

四种分类器的准确率比较：knn的分类效果还不错

KNN 准确率：0.9756
SVM 准确率：0.9867
多项式朴素贝叶斯准确率：0.8844
CART 决策树的准确率为：0.8467