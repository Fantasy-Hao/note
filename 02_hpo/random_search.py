import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 分割数据集为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


# 定义随机搜索函数
def random_search(X_train, y_train, X_val, y_val, model, param_space, iteration_num):
    best_score = -1
    best_params = None
    for i in range(iteration_num):
        # 从参数空间中随机采样一组超参数
        params = {k: v[np.random.randint(len(v))] for k, v in param_space.items()}
        # 训练模型并计算验证集上的准确率
        model.set_params(**params)
        model.fit(X_train, y_train)  # 训练模型
        score = model.score(X_val, y_val)  # 评估模型
        # 更新最优解
        if score > best_score:
            best_score = score
            best_params = params
    return best_score, best_params


# 定义随机森林分类器模型
model = RandomForestClassifier()

# 定义超参数空间
param_space = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [2, 3, 4, 5, 6],
    'max_features': [None, 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# 执行随机搜索
best_score, best_params = random_search(X_train, y_train, X_val, y_val, model, param_space, 100)
print('最佳准确率：', best_score)
print('最佳超参数：', best_params)
