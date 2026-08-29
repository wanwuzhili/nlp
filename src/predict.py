import torch
import re


def predict(net, prefix:str, vocab, seq_len, num_pred=1):
    prefix = re.sub('A-Za-z+', ' ', prefix).strip().lower()
    pre_tokens = list(prefix)
    if len(pre_tokens) < seq_len:
        unks = ['<unk>'] * (seq_len - pre_tokens)
        pre_tokens += unks
    else:
        pre_tokens = pre_tokens[-seq_len:]
    pre_tokens = torch.tensor(vocab.encode(pre_tokens), dtype=torch.long)
    X = pre_tokens.unsqueeze(0) # batch_size=1, [1, T]
    pred = ""
    net.eval()
    for _ in range(num_pred):
        Y_hat = net(X).squeeze(0) # [T, V]
        Y_hat = Y_hat.argmax(-1)
        y_pred = Y_hat[-1]
        pred += vocab.decode(y_pred.item())
        X = torch.concat([X[:, 1:], y_pred.reshape(-1, 1)], dim=1)

    print(prefix + pred)