import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.filter import ExpressionDFilter
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import CSZScoreNorm, DropnaProcessor, ZScoreNorm
from qlib.data.dataset import DatasetH


if __name__ == '__main__':
    provider_uri = "~/dev/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 测试通过D.featrues获取基础数据
    # df = D.features(['SH601216'], ['$open', '$high', '$low', '$close', '$factor'], start_time='2010-01-01',
    #                 end_time='2023-08-08')

    # 测试获取某一只股票的数据基础数据  TODO（股票价格与实际不一致）
    instruments = ['SH600519']
    features = ['$open', '$close', '$high', '$low', '$volume', '$change', '$factor']
    names = ['open', 'close', 'high', 'low', 'volume', 'change', 'factor']

    qdl = QlibDataLoader(config=(features, names))
    df = qdl.load(instruments=instruments, start_time='20200101', end_time='20231007')

    # 测试加载沪深300成分股的10日和30日收盘价指数加权均价
    # start_time = '20190101'
    # end_time = '20200101'
    # market = 'csi300'  # 沪深300股票池代码，在instruments文件夹下有对应的sh000300.txt
    # close_ma = ['$close', 'EMA($close, 10)', 'EMA($close, 30)']  # EMA($close, 10)表示计算close的10日指数加权均线
    # ma_names = ['close', 'EMA10', 'EMA30']
    # qdl_ma = QlibDataLoader(config=(close_ma, ma_names))
    # df = qdl_ma.load(instruments=market, start_time=start_time, end_time=end_time)

    # 测试获取csi100的收益率数据
    # market = 'csi100'  # 沪深300股票池代码，在instruments文件夹下有对应的sh000300.txt
    # close_ma = ['EMA($close, 10)', 'EMA($close, 30)']  # EMA($close, 10)表示计算close的10日指数加权均线
    # ma_names = ['EMA10', 'EMA30']
    # ret = ["Ref($close, -1) / $close - 1"]  # 下一日收益率, Ref($close, -1)表示下一日收盘价
    # ret_name = ['next_ret']
    # qdl_ma_gp = QlibDataLoader(config={'feature': (close_ma, ma_names), 'label': (ret, ret_name)})
    # df = qdl_ma_gp.load(instruments=market, start_time='20190101', end_time='20200101')

    # 测试QlibDataLoader其他参数: filter_pipe   TODO（筛选出的结果有一部分不符合要求）
    # market = 'csi300'  # 沪深300股票池代码，在instruments文件夹下有对应的csi300.txt
    # close_ma = ['EMA($close, 10)', 'EMA($close, 30)']  # EMA($close, 10)表示计算close的10日指数加权均线
    # ma_names = ['EMA10', 'EMA30']
    # # 使用表达式定义过滤规则
    # filter_rule = ExpressionDFilter(rule_expression='EMA($close, 10) > EMA($close, 30)')
    # # 导出数据
    # qdl_fil = QlibDataLoader(config=(close_ma, ma_names), filter_pipe=[filter_rule,])
    # df = qdl_fil.load(instruments=market, start_time='20190101', end_time='20200101')

    # 测试数据预处理函数
    # qdl = QlibDataLoader(config=(['$close/Ref($close, 1)-1'], ['Return']))
    # df = qdl.load(instruments='csi300', start_time='20190101', end_time='20200101')
    # # 是否有空值
    # df.isna().sum()
    # # 原始数据分布
    # df.xs('2019-01-02').hist()
    # # 实例化DataHandler
    # dh = DataHandlerLP(instruments='csi300', start_time='20190101', end_time='20200101',
    #                    learn_processors=[DropnaProcessor(), CSZScoreNorm()],
    #                    data_loader=qdl)
    # # 获取处理后的数据，处理过程为先去空值，再截面标准化。
    # df_hdl = dh.fetch(data_key=DataHandlerLP.DK_L)

    # 测试3种类型的数据预处理方式
    # qdl = QlibDataLoader(config=(['$close/Ref($close, 1)-1'], ['Return']))
    # # 分别定义shared_processors, learn_processors, infer_processors
    # shared_processors = [DropnaProcessor()]
    # learn_processors = [CSZScoreNorm()]
    # infer_processors = [ZScoreNorm(fit_start_time='20190101', fit_end_time='20200101')]
    # dh_pr_test = DataHandlerLP(instruments='csi300',
    #                            start_time='20190101',
    #                            end_time='20200101',
    #                            process_type=DataHandlerLP.PTYPE_I,
    #                            learn_processors=learn_processors,
    #                            shared_processors=shared_processors,
    #                            infer_processors=infer_processors,
    #                            data_loader=qdl)
    # # 原始数据
    # _raw_df = dh_pr_test.fetch(data_key=DataHandlerLP.DK_R)
    #
    # # 处理后的数据
    # _infer_df = dh_pr_test.fetch(data_key=DataHandlerLP.DK_I)
    # _learn_df = dh_pr_test.fetch(data_key=DataHandlerLP.DK_L)

    # 测试DataLoader功能
    # 实例化Data Loader
    # market = 'csi300'  # 沪深300股票池代码，在instruments文件夹下有对应的csi300.txt
    # close_ma = ['EMA($close, 10)', 'EMA($close, 30)']  # EMA($close, 10)表示计算close的10日指数加权均线
    # ma_names = ['EMA10', 'EMA30']
    # ret = ["Ref($close, -1) / $close - 1"]  # 下一日收益率, Ref($close, -1)表示下一日收盘价
    # ret_name = ['next_ret']
    # qdl_ma_gp = QlibDataLoader(config={'feature': (close_ma, ma_names), 'label': (ret, ret_name)})
    #
    # # 实例化Data Handler
    # shared_processors = [DropnaProcessor()]
    # learn_processors = [CSZScoreNorm()]
    # infer_processors = [ZScoreNorm(fit_start_time='20190101', fit_end_time='20200101')]
    #
    # dh_pr_test = DataHandlerLP(instruments=market,
    #                            start_time='20190101',
    #                            end_time='20200101',
    #                            process_type=DataHandlerLP.PTYPE_I,
    #                            learn_processors=learn_processors,
    #                            shared_processors=shared_processors,
    #                            infer_processors=infer_processors,
    #                            data_loader=qdl_ma_gp)
    #
    # ds = DatasetH(handler=dh_pr_test, segments={"train": ('20190101', '20190531'), "test": ('20190601', '20200101')})