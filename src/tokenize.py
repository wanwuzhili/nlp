import collections
import re
import os


def read_file(data_dir):
    data_path = os.path.join(data_dir, 'timemachine.txt')
    with open(data_path, 'r') as f:
        lines = f.readlines()
    
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        self.tokens = tokens if tokens is not None else []
        self.idx_token = ['<unk>']
        if reserved_tokens is not None:
            self.idx_token += reserved_tokens
        self.token_idx = {}
        counter = self.counts(self.tokens)
        self.token_freq = sorted(counter.items(), key=lambda x:x[1], reverse=True)
        for token, freq in self.token_freq:
            if freq < min_freq:
                break
            self.idx_token.append(token)

        for idx, token in enumerate(self.idx_token):
            self.token_idx[token] = idx

    def counts(self, tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]

        return collections.Counter(tokens)

    def __len__(self):
        return len(self.idx_token)

    def encode(self, tokens):
        if type(tokens) not in (list, tuple):
            return self.token_idx.get(tokens, 0)
        
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]

        return [self.token_idx.get(token, 0) for token in tokens]

    def decode(self, idxs):
        if type(idxs) not in (list, tuple):
                return self.idx_token[idxs]
        if len(idxs) == 0 or isinstance(idxs[0], list):
            idxs = [idx for line in idxs for idx in line]

        return [self.idx_token[idx] for idx in idxs]

def load_corpus_time_machine(data_dir='../data/',method='char', min_freq=0,
                             reserved_tokens=None, max_tokens=-1):
    lines = read_file(data_dir)
    tokens = tokenize(lines, token=method)
    vocab = Vocab(tokens,min_freq=min_freq, reserved_tokens=reserved_tokens)
    corpus = vocab.encode(tokens)

    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab