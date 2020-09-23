# encoding=utf-8
from src.seq2seq_tf import Seq2seq
from src.util.logger import setlogger
from src.util.yaml_util import loadyaml
import os
import tensorflow as tf

config = loadyaml('./conf/nlg.yaml')
logger = setlogger(config)

# tf配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8


def test_seq2seq():
    cfg = {
        'logger': logger,
        'train_path': config['train_path'],
        'max_len': 250,
        'batch_size': 2,
        'epoch': 100,
        'loss': 'adam',
        'rate': 0.01,
        'num_units': 64,
        'tf_config': tf_config,
        'model_path': config['seq2seq_model_path'],
        'summary_path': config['seq2seq_summary_path']
    }
    model = Seq2seq(**cfg)
    model.fit()
    model.load(config['seq2seq_predict_path'])
    print(model.predict('你叫什么名字'))
    model.close()


if __name__ == '__main__':
    test_seq2seq()