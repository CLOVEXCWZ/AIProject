""" 对字符型进行编码解码处理
"""

import collections

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
START_TOKEN = '[BOS]'
END_TOKEN = '[EOS]'
MASK_TOKEN = '[MASK]'


class ChartTokenizer(object):
    """ 主要针对字符处理 """

    def __init__(self,
                 vocab,
                 start_token=START_TOKEN,
                 end_token=END_TOKEN,
                 unk_token=UNK_TOKEN,
                 pad_token=PAD_TOKEN):
        """
        正对字符级别，token encoder

        :param vocab: 词典 {char:index, ...,}
        :param start_token: 句子开始标记 默认 '[GO]'
        :param end_token: 句子结束标记 默认 '[EOS]'
        :param unk_token: 未登录标记 默认 '[UNK]'
        :param pad_token: 填充标记 默认 '[PAD]'
        """
        if isinstance(vocab, dict):
            self.vocab = vocab
        elif isinstance(vocab, collections.abc.Iterable):
            self.vocab = {c: i for i, c in enumerate(vocab)}
        else:
            raise ValueError("请检查传入的 vocab 当前只支持 dict、Iterable 类型")

        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 1)
        self.start_token_id = self.vocab.get(self.start_token, 2)
        self.end_token_id = self.vocab.get(self.end_token, 3)

    @staticmethod
    def _truncate(tokens, max_len=None, start=True, end=True):
        """ 截断超出的字符 """
        if max_len is None:
            return
        n = (1 if start else 0) + (1 if end else 0)
        del tokens[max_len - n:]

    def _convert_tokens_to_ids(self, tokens):
        """ token 转成 ids """
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def tokenize(self, text):
        """ 文本处理成 token
        例如: "机器学习" -> ['机', '器', '学', '习']
        """
        tokens = self._tokenize(text)
        return tokens

    def encode(self, text, max_len=None, start=True, end=True):
        """ 编码 """
        tokens = self._tokenize(text)
        self._truncate(tokens, max_len, start, end)
        token_ids = self._convert_tokens_to_ids(tokens)
        if start:
            token_ids = [self.start_token_id] + token_ids
        if end:
            token_ids = token_ids + [self.end_token_id]
        if max_len is not None:
            pad_len = max_len - len(token_ids)
            token_ids += [self.pad_token_id] * pad_len
        return token_ids

    def decode(self, ids, cut_end=True):
        """解码

        :param ids: list id 列表
        :param cut_end: int 截断 end_id 后的token
        :return: list
        """
        ids_new = []
        if cut_end:
            for i in ids:
                if i == self.end_token_id:
                    break
                ids_new.append(i)
        else:
            ids_new = ids
        tokens = [self.vocab_inv.get(i, self.unk_token) for i in ids_new]
        return tokens

    def _tokenize(self, text):
        """ 处理成token (简化处理) """
        spaced = ''
        for ch in text:
            spaced += ch + ' '
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens

    def _word_piece_tokenize(self, word):
        if word in self.vocab:
            return [word]
        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self.vocab:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop
        return tokens

