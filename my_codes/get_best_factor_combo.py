import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import qlib
from qlib.contrib.data.handler import Alpha158, MyFeatureSetStock
from qlib.data import D

# 初始化QLib并指定数据存储路径
provider_uri = 'E:/dev/qlib_data/cn_data_day'
qlib.init(provider_uri=provider_uri)


def find_best_factors(X: pd.DataFrame, y: pd.Series, n_factors: int):
    """
    寻找最佳因子组合
    Parameters
    ----------
    X: pd.DataFrame 特征矩阵
    y: pd.Series 目标变量
    n_factors: int 需要选择的因子数量

    Returns: tuple of (tuple, np.ndarray, float)
    -------
    """
    best_combo = None
    best_r2 = -np.inf
    best_model = None

    # 遍历所有可能的因子组合
    for combo in combinations(X.columns, n_factors):
        X_combo = X[list(combo)]  # 用选定的因子创建新的DataFrame
        model = LinearRegression()
        model.fit(X_combo, y)
        y_pred = model.predict(X_combo)
        r2 = r2_score(y, y_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_combo = combo
            best_model = model

    return best_combo, best_model.coef_, best_r2


if __name__ == '__main__':
    # 获取自定义的因子数据
    handler = MyFeatureSetStock(start_time="2023-11-01",
                                end_time="2023-11-20",
                                instruments='csi300')

    df = handler.fetch()
    print(df.head())

    # 将最后一列作为目标变量 y
    y = df.iloc[:, -1]

    # 将除了最后一列之外的所有列作为特征矩阵 X
    X = df.iloc[:, :-1]

    # 找到最佳的n个因子组合
    best_factors, best_factor_weights, best_r2_value = find_best_factors(X, y, 3)

    print(f"Best factor combination: {best_factors}")
    print(f"Best factor weights: {best_factor_weights}")
    print(f"Best R2 value: {best_r2_value}")