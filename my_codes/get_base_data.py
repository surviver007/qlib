import pandas as pd
import qlib
from qlib.data import D
import plotly.graph_objects as go
from qlib.data.dataset.loader import QlibDataLoader, StaticDataLoader


if __name__ == '__main__':
    # 基础参数
    start_time = '20230101'
    end_time = '20231020'
    # instruments = ['000001.SZ', '000002.SZ']
    fields = ['$open', '$high', '$low', '$close', '$volume', '$amount', 'EMA($close, 10)', 'EMA($close, 30)']
    names = ['open', 'high', 'low', 'close', 'volume', 'amount', 'EMA10', 'EMA30']
    freq = 'day'

    # 初始化QLib并指定数据存储路径
    provider_uri = 'E:/dev/qlib_data/cn_data_day'
    # provider_uri = 'E:/dev/qlib_data/cn_data_1min'

    qlib.init(provider_uri=provider_uri)

    # ==================================================
    # 获取交易日历
    calendar = D.calendar(start_time=start_time, end_time=end_time, freq=freq)
    # 获取股票标的
    instruments = D.instruments(market='all')

    # 获取基础数据(D无法给fields的名字进行命名，dataloader可以)
    df = D.features(
        instruments=instruments,
        fields=fields,
        start_time=start_time,
        end_time=end_time,
        freq=freq
    )

    # ==================================================
    # 通过dataloader获取数据和因子
    # qd = QlibDataLoader(config = (fields, names))
    # data = qd.load(instruments=instruments, start_time=start_time, end_time=end_time)
