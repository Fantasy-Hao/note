from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 定义模型
svm = SVC(kernel='linear', C=100, gamma='auto')

# 定义网格搜索参数范围
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1e-3, 1e-2, 1e-1, 1],
}

# 创建网格搜索对象
grid_search = GridSearchCV(svm, param_grid, cv=5)

# 对数据进行网格搜索
grid_search.fit(X, y)

# 输出最佳参数组合和对应的得分
print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)
