import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP

data_uri = "~/dev/qlib_data/cn_data"
qlib.init(provider_uri=data_uri, region=REG_CN)


class MyFeatureSet(DataHandlerLP):
    """实现自定义特征集，MACD、RSI"""
    def __init__(self,
                 instruments="csi300",
                 start_time=None,
                 end_time=None,
                 freq="day",
                 infer_processors=[],
                 learn_processors=[],
                 fit_start_time=None,
                 fit_end_time=None,
                 process_type=DataHandlerLP.PTYPE_A,
                 filter_pipe=None,
                 inst_processor=None,
                 **kwargs,
                 ):

        data_loader = {
            "class": "QlibDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": {
                    "feature": self._get_feature_config(),
                    "label": kwargs.get("label", self._get_label_config()),  # label可以自定义，也可以使用初始化时候的设置
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processor,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
        )

    @staticmethod
    def _get_feature_config():
        MACD = '(EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close'
        RSI = ('100 - 100 / (1+(Sum(Greater($close-Ref($close, 1),0), 14)/Count(($close-Ref($close, 1))>0, 14))/(Sum('
               'Abs(Greater(Ref($close, 1)-$close,0)), 14)/Count(($close-Ref($close, 1))<0, 14)))')

        return [MACD, RSI], ['MACD', 'RSI']

    @staticmethod
    def _get_label_config():
        return ["Ref($close, -1)/$close - 1"], ["Label"]


class MyFeatureSetStock(DataHandlerLP):
    """实现自定义特征集，包含中金日频量价因子"""
    def __init__(self,
                 instruments="csi300",
                 start_time=None,
                 end_time=None,
                 freq="day",
                 infer_processors=[],
                 learn_processors=[],
                 fit_start_time=None,
                 fit_end_time=None,
                 process_type=DataHandlerLP.PTYPE_A,
                 filter_pipe=None,
                 inst_processor=None,
                 **kwargs,
                 ):

        data_loader = {
            "class": "QlibDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": {
                    "feature": self._get_feature_config(),
                    "label": kwargs.get("label", self._get_label_config()),  # label可以自定义，也可以使用初始化时候的设置
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processor,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
        )

    @staticmethod
    def _get_feature_config():
        features = []
        names = []

        # =============================
        # 基础因子
        features += ['$open', '$close', '$high', '$low', '$volume']
        names += ['open', 'close', 'high', 'low', 'volume']

        # =============================
        # 动量因子
        # 1个月隔夜动量
        features += ['Sum(($open / Ref($close, 1) - 1), 20)']
        names += ['mmt_overnight_M']

        # 1年隔夜动量
        features += ['Sum(($open - Ref($close, 1)), 252) - Sum(($open - Ref($close, 1)), 20)']
        names += ['mmt_overnight_A']

        # 1个月横截面rank动量  todo 缺少截面rank算子
        # features += ['Mean(Rank(($close / Ref($close, 1) - 1)), 20)']
        # names += ['mmt_sec_rank_M']

        # 1年横截面rank动量  todo 无横截面rank算子
        # mmt_sec_rank_A = MA(RANK(PCT_CHANGE(close, 1)), 252)

        # 1个月路径调整动量
        features += ['($close / Ref($close, 20) - 1) / Sum(Abs(($close / Ref($close, 1) - 1)), 20)']
        names += ['mmt_route_M']

        # 1年路径调整动量
        features += ['($close / Ref($close, 252) - 1) / Sum(Abs(($close / Ref($close, 1) - 1)), 252)']
        names += ['mmt_route_A']

        # 1个月日内动量
        features += ['Sum(($close / $open - 1), 20)']
        names += ['mmt_intraday_M']

        # 1年日内动量
        features += ['Sum(($close / $open - 1), 252) - Sum(($close / $open - 1), 20)']
        names += ['mmt_intraday_A']

        # 1个月时序rank动量
        features += ['Mean(Rank($close, 252), 20)']
        names += ['mmt_time_rank_M']

        # 1个月收益率
        features += ['($close / Ref($close, 20) - 1)']
        names += ['mmt_normal_M']

        # 1年收益率
        features += ['($close / Ref($close, 252) - 1) - ($close / Ref($close, 20) - 1)']
        names += ['mmt_normal_A']

        # 1个月信息离散度动量（count函数不可用，更换成sum+if实现，但是该为nan的显示为0）
        features += ['Sum(If($close > Ref($close, 1), 1, 0), 20) / 20 - Sum(If($close < Ref($close, 1), 1, 0), 20) / 20']
        names += ['mmt_discrete_M']

        # 1年信息离散度动量（count函数不可用，更换成sum+if实现，但是该为nan的显示为0）
        features += ['Sum(If($close > Ref($close, 1), 1, 0), 252) / 252 - Sum(If($close < Ref($close, 1), 1, 0), 252) / 252']
        names += ['mmt_discrete_A']

        # 1个月振幅调整动量  todo 无法实现
        # mmt_range_M = SK_SUM_UP_20(high / low, PCT_CHANGE(close, 1), 22) - SK_SUM_DOWN_20(high / low, PCT_CHANGE(close, 1), 22)

        # 1年振幅调整动量  todo 无法实现
        # mmt_range_A = SK_SUM_UP_20(high / low, PCT_CHANGE(close, 1), 252) - SK_SUM_DOWN_20(high / low, PCT_CHANGE(close, 1), 252)

        # 近1年最高价日期距今的天数 todo 待确认IdxMax的含义
        features += ['251 - IdxMax($high, 252)']
        names += ['mmt_highest_days_A']

        # 相对均价的1个月收益率
        features += ['$close / Mean($close, 20)']
        names += ['mmt_avg_M']

        # 相对均价的1年收益率
        features += ['Ref($close, 20) / Mean($close, 252)']
        names += ['mmt_avg_A']

        # 滚动3个月历史股价分位数
        features += ['Rank($close, 60)']
        names += ['close_quantile_3m']

        # =============================
        # 波动率因子
        # 1个月波动率
        features += ['Std(($close / Ref($close, 1) - 1), 20)']
        names += ['vol_std_1M']

        # 1个月日内振幅
        features += ['Mean($high / $low, 20)']
        names += ['vol_highlow_avg_1M']

        # 1个月日内振幅标准差
        features += ['Std($high / $low, 20)']
        names += ['vol_highlow_std_1M']

        # 1个月上行波动率
        features += ['Std(If(($close / Ref($close, 1) - 1) > 0, ($close / Ref($close, 1) - 1), np.NAN), 20)']
        names += ['vol_up_std_1M']

        # 1个月下行波动率
        features += ['Std(If(($close / Ref($close, 1) - 1) < 0, ($close / Ref($close, 1) - 1), np.NAN), 20)']
        names += ['vol_down_std_1M']

        # 1个月上影线标准差
        features += ['Std(($high - Greater($open, $close)) / $high, 20)']
        names += ['vol_upshadow_std_1M']

        # 1个月下影线标准羞
        features += ['Std((Less($open, $close) - $low) / $low, 20)']
        names += ['vol_downshadow_std_1M']

        # 1个月上影线均值
        features += ['Mean(($high - Greater($open, $close)) / $high, 20)']
        names += ['vol_upshadow_avg_1M']

        # 1个月下影线均值
        features += ['Mean((Less($open, $close) - $low) / $low, 20)']
        names += ['vol_downshadow_avg_1M']

        # 1个月威廉上影线标准差
        features += ['Std(($high - $close) / $high, 20)']
        names += ['vol_w_upshadow_std_1M']

        # 1个月威廉下影线标准差
        features += ['Std(($close - $low) / $low, 20)']
        names += ['vol_w_downshadow_std_1M']

        # 1个月威廉上影线均值
        features += ['Mean(($high - $close) / $high, 20)']
        names += ['vol_w_upshadow_avg_1M']

        # 1个月威廉下影线均值
        features += ['Mean(($close - $low) / $low, 20)']
        names += ['vol_w_downshadow_avg_1M']

        # 3个月波动率
        features += ['Std(($close / Ref($close, 1) - 1), 60)']
        names += ['vol_std_3M']

        # 3个月日内振幅
        features += ['Mean($high / $low, 60)']
        names += ['vol_highlow_avg_3M']

        # 3个月日内振幅标准差
        features += ['Std($high / $low, 60)']
        names += ['vol_highlow_std_3M']

        # 3个月上行波动率
        features += ['Std(If(($close / Ref($close, 1) - 1) > 0, ($close / Ref($close, 1) - 1), np.NAN), 60)']
        names += ['vol_up_std_3M']

        # 3个月下行波动率
        features += ['Std(If(($close / Ref($close, 1) - 1) < 0, ($close / Ref($close, 1) - 1), np.NAN), 60)']
        names += ['vol_down_std_3M']

        # 3个月上影线标准差
        features += ['Std(($high - Greater($open, $close)) / $high, 60)']
        names += ['vol_upshadow_std_3M']

        # 3个月下影线标准羞
        features += ['Std((Less($open, $close) - $low) / $low, 60)']
        names += ['vol_downshadow_std_3M']

        # 3个月上影线均值
        features += ['Mean(($high - Greater($open, $close)) / $high, 60)']
        names += ['vol_upshadow_avg_3M']

        # 3个月下影线均值
        features += ['Mean((Less($open, $close) - $low) / $low, 60)']
        names += ['vol_downshadow_avg_3M']

        # 3个月威廉上影线标准差
        features += ['Std(($high - $close) / $high, 60)']
        names += ['vol_w_upshadow_std_3M']

        # 3个月威廉下影线标准差
        features += ['Std(($close - $low) / $low, 60)']
        names += ['vol_w_downshadow_std_3M']

        # 3个月威廉上影线均值
        features += ['Mean(($high - $close) / $high, 60)']
        names += ['vol_w_upshadow_avg_3M']

        # 3个月威廉下影线均值
        features += ['Mean(($close - $low) / $low, 60)']
        names += ['vol_w_downshadow_avg_3M']

        # 6个月波动率
        features += ['Std(($close / Ref($close, 1) - 1), 120)']
        names += ['vol_std_6M']

        # 6个月日内振幅
        features += ['Mean($high / $low, 120)']
        names += ['vol_highlow_avg_6M']

        # 6个月日内振幅标准差
        features += ['Std($high / $low, 120)']
        names += ['vol_highlow_std_6M']

        # 6个月上行波动率
        features += ['Std(If(($close / Ref($close, 1) - 1) > 0, ($close / Ref($close, 1) - 1), np.NAN), 120)']
        names += ['vol_up_std_6M']

        # 6个月下行波动率
        features += ['Std(If(($close / Ref($close, 1) - 1) < 0, ($close / Ref($close, 1) - 1), np.NAN), 120)']
        names += ['vol_down_std_6M']

        # 6个月上影线标准差
        features += ['Std(($high - Greater($open, $close)) / $high, 120)']
        names += ['vol_upshadow_std_6M']

        # 6个月下影线标准羞
        features += ['Std((Less($open, $close) - $low) / $low, 120)']
        names += ['vol_downshadow_std_6M']

        # 6个月上影线均值
        features += ['Mean(($high - Greater($open, $close)) / $high, 120)']
        names += ['vol_upshadow_avg_6M']

        # 6个月下影线均值
        features += ['Mean((Less($open, $close) - $low) / $low, 120)']
        names += ['vol_downshadow_avg_6M']

        # 6个月威廉上影线标准差
        features += ['Std(($high - $close) / $high, 120)']
        names += ['vol_w_upshadow_std_6M']

        # 6个月威廉下影线标准差
        features += ['Std(($close - $low) / $low, 120)']
        names += ['vol_w_downshadow_std_6M']

        # 6个月威廉上影线均值
        features += ['Mean(($high - $close) / $high, 120)']
        names += ['vol_w_upshadow_avg_6M']

        # 6个月威廉下影线均值
        features += ['Mean(($close - $low) / $low, 120)']
        names += ['vol_w_downshadow_avg_6M']

        # =============================
        # 量价关系因子
        # 换手率与价格相关性因子(量价同步)   todo（没有capital字段）
        # corr_price_turn_1M = CORR(volume / capital, close, 20)

        # =============================
        # 流动性因子
        # 1个月Amihud非流动因子   todo(没有total_turnover字段)
        # features = ['Mean(($close / Ref($close, 1) - 1) * 1000000000 / total_turnover, 20)']
        # names += ['liq_amihud_avg_1M']


        return features, names

    @staticmethod
    def _get_label_config():
        return ["Ref($close, -1)/$close - 1"], ["Label"]


if __name__ == '__main__':
    # 测试简单的获取数据的方法，先配置config，然后调用load
    # qdl_config = {
    #     "class": "QlibDataLoader",
    #     "module_path": "qlib.data.dataset.loader",
    #     "kwargs": {
    #         "config": {
    #             "feature": (['EMA($close, 10)', 'EMA($close, 30)'], ['EMA10', 'EMA30']),
    #             "label": (['Ref($close, -1) / $close - 1', ], ['RET_1', ]),
    #         },
    #         "freq": 'day',
    #     },
    # }
    #
    # qdl = init_instance_by_config(qdl_config)
    #
    # market = 'csi300'
    # df = qdl.load(instruments=market, start_time='20190101', end_time='20200101')

    # 通过自定义类的方式调用feature和label数据
    instruments = 'csi300'
    start_time = '20180101'
    end_time = '20180201'
    my_feature = MyFeatureSetStock(instruments=instruments, start_time=start_time, end_time=end_time)
    df = my_feature.fetch()
    df_feature = my_feature.fetch(col_set='feature')
    df_label = my_feature.fetch(col_set='label')
