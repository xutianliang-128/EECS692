from nltk.tokenize import word_tokenize
#from nltk.translate.bleu_score import sentence_bleu

import fasttext
import pkg_resources
import kenlm
import math
import torch
import numpy as np

from transformers import (AutoTokenizer)
from transformers.modeling_outputs import SequenceClassifierOutput
from evaluator.huggingface_clf import DistilBertForMultilabelSequenceClassification




class Evaluator(object):

    def __init__(self):
        resource_package = __name__

        yelp_acc_path = 'acc_yelp.bin'
        yelp_ppl_path = 'ppl_yelp.binary'
        #yelp_ref0_path = 'yelp.refs.0'
        #yelp_ref1_path = 'yelp.refs.1'
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        #yelp_acc_file = pkg_resources.resource_stream(resource_package, yelp_acc_path)
        yelp_ppl_file = pkg_resources.resource_stream(resource_package, yelp_ppl_path)
        #yelp_ref0_file = pkg_resources.resource_stream(resource_package, yelp_ref0_path)
        #yelp_ref1_file = pkg_resources.resource_stream(resource_package, yelp_ref1_path)
        
        #self.yelp_ref = []
        """ with open(yelp_ref0_file.name, 'r') as fin:
            self.yelp_ref.append(fin.readlines())
        with open(yelp_ref1_file.name, 'r') as fin:
            self.yelp_ref.append(fin.readlines()) """
        #self.classifier_yelp = fasttext.load_model(yelp_acc_file.name)
        self.classifier_yelp = DistilBertForMultilabelSequenceClassification.from_pretrained(
            "bhadresh-savani/distilbert-base-uncased-go-emotion")
        self.yelp_ppl_model = kenlm.Model(yelp_ppl_file.name)
        
    def acc_sentence(self, text_transfered, style_origin):
        text_transfered = ' '.join(word_tokenize(text_transfered.lower().strip()))
        if text_transfered == '':
            return False
        inputs = self.tokenizer(text_transfered, return_tensors="pt")
        logits = self.classifier_yelp(**inputs).logits

        label = logits.argmax().item()

        return label == style_origin

    def acc_set(self, texts, styles_origin):
        assert len(texts) == len(styles_origin), 'Size of inputs does not match!'
        count = 0
        for text, style in zip(texts, styles_origin):
            if self.acc_sentence(text, style):
                count += 1
        return count / len(texts)

    """     def nltk_bleu(self, texts_origin, text_transfered):
        texts_origin = [word_tokenize(text_origin.lower().strip()) for text_origin in texts_origin]
        text_transfered = word_tokenize(text_transfered.lower().strip())
        return sentence_bleu(texts_origin, text_transfered) * 100

    def self_bleu_b(self, texts_origin, texts_transfered):
        assert len(texts_origin) == len(texts_transfered), 'Size of inputs does not match!'
        sum = 0
        n = len(texts_origin)
        for x, y in zip(texts_origin, texts_transfered):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def yelp_ref_bleu_0(self, texts_neg2pos):
        assert len(texts_neg2pos) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 500
        for x, y in zip(self.yelp_ref[0], texts_neg2pos):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def yelp_ref_bleu_1(self, texts_pos2neg):
        assert len(texts_pos2neg) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 500
        for x, y in zip(self.yelp_ref[1], texts_pos2neg):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def yelp_ref_bleu(self, texts_neg2pos, texts_pos2neg):
        assert len(texts_neg2pos) == 500, 'Size of input differs from human reference file(500)!'
        assert len(texts_pos2neg) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 1000
        for x, y in zip(self.yelp_ref[0] + self.yelp_ref[1], texts_neg2pos + texts_pos2neg):
            sum += self.nltk_bleu([x], y)
        return sum / n """

    
    def yelp_ppl(self, texts_transfered):
        texts_transfered = [' '.join(word_tokenize(itm.lower().strip())) for itm in texts_transfered]
        sum = 0
        words = []
        length = 0
        for i, line in enumerate(texts_transfered):
            words += [word for word in line.split()]
            #if len(line.split()) == 0 : length += 1
            length += len(line.split())
            score = self.yelp_ppl_model.score(line)
            sum += score
        return math.pow(10, -sum / length)

    
