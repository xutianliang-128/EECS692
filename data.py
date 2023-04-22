import time
import numpy as np
import torchtext
from torchtext import data

from utils import tensor2text

import spacy

import torch
class Config():
    data_path = './goemotions-data/'
    log_dir = 'runs/exp'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    max_length = 16
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 29 # Should be 3?
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 29
    num_layers = 4
    batch_size = 64
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 500
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0

class DatasetIterator(object):
    def __init__(self, iter):
        self.iter = iter

    def __iter__(self):
        #return iter(self.iter)
        #for batch_pos in iter(self.iter):
        #    print(batch_pos.text)
        for _ in self.iter:
            yield _

def isolate_class(classes : str):
    return int(classes.split(',')[0])

def load_dataset(config, train="train.tsv",
                 dev='dev.tsv',
                 test='test.tsv'):

    root = config.data_path
    TEXT = data.Field(batch_first=True, eos_token='<eos>', lower=True, tokenize="spacy", tokenizer_language="en_core_web_sm")
    process_classes = data.Pipeline(convert_token=isolate_class)
    STYLE = data.Field(sequential=False, use_vocab=False, preprocessing=process_classes, is_target=True)
    #ID = data.Field(sequential=False, use_vocab=False)

    
    # Using sentiment_dict could be non-deterministic for some texts.
    dataset_fn = lambda name: data.TabularDataset( # do we want to use the same train/test split as GoEmotions?
        path=root + name,
        format='tsv',
        fields=[('text', TEXT), ('style', STYLE), ('id', None)]
    )

    train_set = dataset_fn(train)
    dev_set = dataset_fn(dev)
    test_set = dataset_fn(test)

    TEXT.build_vocab(train_set, min_freq=config.min_freq)

    if config.load_pretrained_embed:
        start = time.time()
        
        vectors=torchtext.vocab.GloVe('6B', dim=config.embed_size, cache=config.pretrained_embed_path)
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print('vectors', TEXT.vocab.vectors.size())
        
        print('load embedding took {:.2f} s.'.format(time.time() - start))

    vocab = TEXT.vocab
        
    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=config.device
    )

    train_iter = dataiter_fn(train_set, True)
    dev_iter = dataiter_fn(dev_set, False)
    test_iter = dataiter_fn(test_set, False)

    train_iters = DatasetIterator(train_iter)
    dev_iters = DatasetIterator(dev_iter)
    test_iters = DatasetIterator(test_iter)
    
    return train_iters, dev_iters, test_iters, vocab


if __name__ == '__main__':
    config = Config()
    train_iter, _, _, vocab = load_dataset(config)
    print(len(vocab))
    for batch in train_iter:
        text = tensor2text(vocab, batch)
        print('\n'.join(text))
        print(batch.label)
        break
