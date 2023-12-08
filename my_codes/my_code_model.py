import qlib
from qlib.constant import REG_CN

from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import TSDatasetH
from qlib.contrib.model.pytorch_alstm_ts import ALSTM

data_uri = "~/dev/qlib_data/cn_data"
qlib.init(provider_uri=data_uri, region=REG_CN)


if __name__ == '__main__':
    # 配置基础参数
    train_period = ("2017-01-01", "2017-12-31")
    valid_period = ("2018-01-01", "2018-12-31")
    test_period = ("2019-01-01", "2019-03-01")
    model_save_path = "./checkpoints/"   # 模型存储路径

    # 定义数据集
    dh = Alpha158(instruments='csi500',
                  start_time=train_period[0],
                  end_time=test_period[1],
                  infer_processors={}
                  )

    # 设置dataset
    ds = TSDatasetH(handler=dh,
                    step_len=40,  # 时间步数
                    segments={"train": train_period,
                              "valid": valid_period,
                              "test": test_period})

    # 配置模型
    model = ALSTM(d_feat=158,
                  metric='loss',
                  rnn_type='GRU',
                  batch_size=800,
                  early_stop=10)

    # 获取数据
    # df = dh.fetch()

    # 模型训练, 使用fit方法
    model.fit(dataset=ds,
              save_path=model_save_path)