import torch
import yaml

from src.tokenize import load_corpus_time_machine
from src.dataset import load_data
import src.model as m
import src.trainer as t
from src.predict import predict


# load configs
with open('./config.yaml', 'r') as f:
    configs = yaml.safe_load(f)

# load data
corpus, vocab = load_corpus_time_machine(
    data_dir=configs['data_dir'], method=configs['token_type']
)
train_iter, val_iter = load_data(
    batch_size=configs['batch_size'], corpus=corpus,
    seq_len=configs['seq_len'], num_workers=configs['nworkers']
)

# load model
vocab_size = len(vocab)
if configs['model_type'] == 't':
    net = m.TransformerDecoder(
        vocab_size=vocab_size,seq_len=configs['seq_len'], d_model=configs['d_model'],
        num_heads=configs['num_heads']
    )
elif configs['model_type'] == 'g':
    net = m.GRUDecoder(
        vocab_size, d_model=configs['d_model'],
        hidden_size=configs['hidden_size'], num_layers=configs['num_layers']
    )
else:
    net = m.RNNDecoder(
        vocab_size, d_model=configs['d_model'],
        hidden_size=configs['hidden_size'], num_layers=configs['num_layers']
    )

# train
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_ls, val_ls = t.train(
    net, train_iter, val_iter, lr=configs['lr'], wd=configs['wd'],
    num_epochs=configs['num_epochs'], device=device
)

t.plot_loss(train_ls, val_ls, save=configs['loss_path'])
# torch.save(net.state_dict(), configs['model_path'])

# predict
net = net.to(torch.device('cpu'))
prefix1 = 'the time machine'
prefix2 = 'hello'
predict(net, prefix=prefix1, vocab=vocab, seq_len=configs['seq_len'],
        num_pred=configs['seq_len']
)
predict(net, prefix=prefix2, vocab=vocab, seq_len=configs['seq_len'],
        num_pred=configs['seq_len']
)