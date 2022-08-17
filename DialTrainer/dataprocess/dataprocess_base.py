""" 数据处理
1.读取文本数据
2.文本数据分样本
3.将文本数据转化成ids
4.将文本数据存储起来（以pkl格式存储）

dialogue

"""

import pickle
import json


class DataProcessBase(object):
    def __int__(self):
        pass

    def fit(self):
        # 将数据处理，并存储在本地
        raise NotImplementedError('请在子类中重写 fit 方法')

    def load_data(self):
        raise NotImplementedError('请在子类中重写 load_data 方法')

    @classmethod
    def save_to_pkl(cls, obj, file_path: str):
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

    @classmethod
    def save_text_to_file(cls, texts, file_path: str):
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(texts)

    @classmethod
    def save_to_json(cls, obj, file_path):
        with open(file_path, 'w') as f:
            json.dump(obj, f)

    @classmethod
    def read_pkl(cls, file_path: str):
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        return obj

    @classmethod
    def read_text(cls, file_path: str, encoding='utf-8'):
        with open(file_path, "r", encoding=encoding) as f:
            texts = f.read()
        return texts

    @classmethod
    def read_json(cls, file_path: str):
        with open(file_path, 'r') as f:
            obj = json.load(f)
        return obj









