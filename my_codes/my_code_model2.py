import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config

from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import TSDatasetH
from qlib.contrib.model.pytorch_alstm_ts import ALSTM

data_uri = "~/dev/qlib_data/cn_data"
qlib.init(provider_uri=data_uri, region=REG_CN)


if __name__ == '__main__':
    """通过配置文件config进行数据获取，数据预处理，特种工程和模型训练"""
    ds_config = {'class': 'TSDatasetH',
                 'module_path': 'qlib.data.dataset',
                 'kwargs': {
                     'handler': {
                         'class': 'Alpha158',
                         'module_path': 'qlib.contrib.data.handler',
                         'kwargs': {'start_time': "2018-01-01",
                                    'end_time': "2020-03-01",
                                    'fit_start_time': "2018-01-01",
                                    'fit_end_time': "2019-12-31",
                                    'instruments': 'csi300',
                                    # 与之前的示例相比，这里新增了infer_processors和learn_processors
                                    'infer_processors': [
                                        {'class': 'RobustZScoreNorm',
                                         'kwargs': {'fields_group': 'feature', 'clip_outlier': True}},
                                        {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}],
                                    'learn_processors': [{'class': 'DropnaLabel'},
                                                         # 对预测的目标进行截面排序处理
                                                         {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}],
                                    # 预测的目标
                                    'label': ['Ref($close, -1) / $close - 1']}},

                     'segments': {'train': ["2018-01-01", "2018-12-31"],
                                  'valid': ["2019-01-01", "2019-12-31"],
                                  'test': ["2020-01-01", "2020-03-01"]},
                     'step_len': 40}}

    model_config = {'class': 'ALSTM',
                    'module_path': 'qlib.contrib.model.pytorch_alstm_ts',
                    'kwargs': {
                        'd_feat': 158,
                        'hidden_size': 64,
                        'num_layers': 2,
                        'dropout': 0.0,
                        'n_epochs': 10,   # 为了测试速度改成1（200）
                        'lr': 1e-3,
                        'early_stop': 10,
                        'batch_size': 800,
                        'metric': 'loss',
                        'loss': 'mse',
                        'n_jobs': 10,   # 之前是20
                        'GPU': 0,
                        'rnn_type': 'GRU'
                    }}

    # 实例化数据集及模型
    ds = init_instance_by_config(ds_config)

    model = init_instance_by_config(model_config)

    # 模型训练
    model.fit(dataset=ds)