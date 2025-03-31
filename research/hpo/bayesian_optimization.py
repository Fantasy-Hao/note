import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# 定义目标函数
def f(x):
    return np.sin(5 * x) + np.cos(x)


# 定义高斯过程回归模型
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)


# 定义贝叶斯优化函数
def bayesian_optimization(X_train, y_train, X_test):
    # 训练高斯过程回归模型
    gpr.fit(X_train, y_train)
    # 计算测试集的预测值和方差
    y_pred, sigma = gpr.predict(X_test, return_std=True)
    # 计算期望改进（Expected Improvement）
    gap = y_pred - f(X_test)
    improvement = (gap + np.sqrt(sigma ** 2 + 1e-6) * np.abs(gap).mean()) * 0.5

    # 计算高斯过程回归模型的超参数
    # 定义优化目标函数
    def objective(params):
        gpr.kernel_.theta = params
        return -gpr.log_marginal_likelihood()

    # 初始化参数
    initial_params = gpr.kernel_.theta
    # 优化超参数
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=gpr.kernel_.bounds)
    hyperparameters = result.x
    # 输出最优超参数和对应的期望改进值
    return hyperparameters, improvement.max()


# 定义贝叶斯优化的迭代次数和采样点数量
n_iter = 20
n_samples = 5

# 进行贝叶斯优化
results = []
for i in range(n_iter):
    # 在定义域内随机采样n_samples个点
    X_train = np.random.uniform(-2 * np.pi, 2 * np.pi, (n_samples, 1))
    y_train = f(X_train)
    # 进行贝叶斯优化并记录最优超参数和对应的期望改进值
    result = bayesian_optimization(X_train, y_train, X_test=np.random.uniform(-2 * np.pi, 2 * np.pi, (100, 1)))
    results.append(result)
    print('Iter: {}, Hyperparameters: {}, Expected Improvement: {:.4f}'.format(i, result[0], result[1]))
