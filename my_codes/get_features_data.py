import pandas as pd
import qlib
from qlib.contrib.data.handler import Alpha158, MyFeatureSetStock
from qlib.data import D

# 初始化QLib并指定数据存储路径
provider_uri = 'E:/dev/qlib_data/cn_data_day'
qlib.init(provider_uri=provider_uri)


if __name__ == '__main__':
    # 获取全部股票标的
    instruments = D.instruments(market='all')

    # 获取自定义的因子数据
    handler = MyFeatureSetStock(start_time="2019-01-01",
                                end_time="2023-11-20",
                                instruments=instruments)
    df = handler.fetch()

    # 计算因子间相关性
    # 使用 groupby 方法按日期对数据进行分组
    grouped = df.groupby(level='datetime')

    # 在每个日期下计算因子的相关性，并将结果存储在字典中
    correlation_dict = {datetime: group.corr(method='spearman') for datetime, group in grouped}

    # 将字典转换为 DataFrame
    correlation_df = pd.concat(correlation_dict, names=['datetime', 'feature'])
    print(correlation_df.head())

    # 按featrue对相关性进行平均
    average_correlation = correlation_df.groupby('feature').mean()
    print(average_correlation.head())

    # 对结果进行排序
    average_correlation.sort_index(axis=0, inplace=True)
    average_correlation.sort_index(axis=1, inplace=True)

