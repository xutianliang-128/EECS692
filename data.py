import time
import numpy as np
import torchtext
from torchtext import data

from utils import tensor2text

import spacy

from random import randrange # for picking a random style from the ground truths
import random
random.seed(692)

import torch

class DatasetIterator(object):
    def __init__(self, iter):
        self.iter = iter

    def __iter__(self):
        for _ in self.iter:
            yield _

def isolate_class(classes : str):
    styles = classes.split(',')
    rand_style = styles[randrange(len(styles))]
    return int(rand_style)

def load_dataset(config, train="train.tsv",
                 dev='dev.tsv',
                 test='test.tsv'):

    root = config.data_path
    TEXT = data.Field(batch_first=True, eos_token='<eos>', lower=True, tokenize="spacy", tokenizer_language="en_core_web_sm", fix_length=config.max_length)
    process_classes = data.Pipeline(convert_token=isolate_class)
    STYLE = data.Field(sequential=False, use_vocab=False, preprocessing=process_classes, is_target=True)
    
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
