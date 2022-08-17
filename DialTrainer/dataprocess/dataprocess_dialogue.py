import os
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

from . dataprocess_base import DataProcessBase


SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


class DialogueDataset(Dataset):
    def __init__(self, data, tokenizer, max_history=15, batch_first=True, lm_labels=True):
        """
        max_history: 历史前N句
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.lm_labels:
            history = self.data[index][-2 * self.max_history:-1]
            resposne = self.data[index][-1]
        else:
            history = self.data[index][-2 * self.max_history:-1]
            resposne = []
        return self.process(history, resposne)

    def process(self, history, resposne, with_eos=True):
        """
        history: list 历史对话 [[sentence 1 token ids], [sentence 1 token ids], ...]
                    例如 [[23, 34, 54], [345, 234, 12]]
        resposne: list 回复  [32, 43, 54]
        with_eos：是否加入eos

        input_ids: [CLS_id] + [speaker1 id] + [sentence 1 token ids] + [speaker2 id] + [sentence 2 token ids]
                    + [speaker1 id] + [resposne ids] + [SEP id]
        token_type_ids: 略
        lm_labels: [-1] ....... [-1]  + [resposne ids] + [SEP id]
        """
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

        sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
        # 每一句前面加上分割符
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))  # 所有拼接到一起
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels


class DialogueDataProcess(DataProcessBase):
    def __init__(self, read_path, tokenizer, save_path, lm_labels=True):
        super(DialogueDataProcess, self).__init__()
        self.read_path = read_path
        self.tokenizer = tokenizer
        self.save_path = save_path
        self.lm_labels = lm_labels
        if not os.path.exists(self.read_path):
            raise ValueError(f"数据源文件不存在 请检查 :{self.read_path}")

    def fit(self):
        raise NotImplementedError('请在子类中重写 fit 方法')

    def _load_data_(self,
                    to_dataset=False,
                    to_dataloader=False,
                    batch_size=32,  # data loader 设置
                    num_workers=0,
                    ):
        if not os.path.exists(self.save_path):
            self.fit()
        dialogue_all = self.read_pkl(self.save_path)
        dialogue_train, dialogue_val = dialogue_all["trainer"], dialogue_all["valid"]
        if to_dataset or to_dataloader:
            dialogue_train = DialogueDataset(dialogue_train, self.tokenizer, lm_labels=self.lm_labels)
            dialogue_val = DialogueDataset(dialogue_val, self.tokenizer, lm_labels=self.lm_labels)
        if to_dataloader:
            dialogue_train = DataLoader(dialogue_train,
                                        collate_fn=dialogue_train.collate,
                                        num_workers=num_workers,
                                        batch_size=batch_size,
                                        shuffle=True)
            dialogue_val = DataLoader(dialogue_val,
                                      collate_fn=dialogue_val.collate,
                                      num_workers=num_workers,
                                      batch_size=batch_size,
                                      shuffle=True)
        return dialogue_train, dialogue_val

    def load_data(self):
        return self._load_data_(to_dataset=False, to_dataloader=False)

    def load_dataset(self):
        return self._load_data_(to_dataset=True)

    def load_dataloader(self, batch_size=32, num_workers=0):
        """ 返回含 input_ids, token_type_ids, lm_labels 的数据

        使用示例子:
        trainer, val = xxx.load_dataloader()
        for batch in trainer:
            input_ids, token_type_ids, lm_labels = tuple(input_tensor.to(device) for input_tensor in batch)

        :param batch_size: 批次大小
        :param num_workers:
        :return:
        """
        return self._load_data_(to_dataloader=True,
                                batch_size=batch_size,
                                num_workers=num_workers)


class ChitDialogueDataProcess(DialogueDataProcess):
    def __init__(self, read_path, tokenizer, save_path, lm_labels=True):
        super(ChitDialogueDataProcess, self).__init__(read_path=read_path,
                                                      tokenizer=tokenizer,
                                                      save_path=save_path,
                                                      lm_labels=lm_labels)

    def _read_samples_(self):
        """ 读取全部样本 """
        texts = self.read_text(self.read_path)
        if "\r\n" in texts:
            samples = texts.split("\r\n\r\n")
        else:
            samples = texts.split("\n\n")
        return samples

    def fit(self, val_num=8000):
        samples = self._read_samples_()
        val_num = min(val_num, int(len(samples) * 0.1))
        dialogue_list = []
        for index, dialogue in enumerate(tqdm(samples)):
            dialogue = str(dialogue).replace("\r\n", "\n")
            utterances = dialogue.split("\n")
            a_sample = []
            for utterance in utterances:
                a_sample.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utterance)))
            dialogue_list.append(a_sample)
        dialogue_train = dialogue_list[val_num:]
        dialogue_val = dialogue_list[:val_num]
        dialogue_all = {'trainer': dialogue_train, 'valid': dialogue_val}
        self.save_to_pkl(dialogue_all, self.save_path)


class LCCCDialogueDataProcess(DialogueDataProcess):
    def __init__(self, read_path, tokenizer, save_path, lm_labels=True):
        super(LCCCDialogueDataProcess, self).__init__(read_path=read_path,
                                                      tokenizer=tokenizer,
                                                      save_path=save_path,
                                                      lm_labels=lm_labels)

    def read_samples(self):
        """ 读取全部样本 """
        datas = self.read_json(self.read_path)
        samples = [{k: v[:5]} for k, v in datas.items()]
        return samples

    def fit(self):
        datas = self.read_json(self.read_path)

        def tokenize(obj):
            if isinstance(obj, str):
                return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        datas = tokenize(datas)
        self.save_to_pkl(datas, self.save_path)

