#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce

import GrobalParament
import utils
from gensim.models import word2vec

#训练word2vec
def train(sentences, model_out_put_path):
    print("开始训练")
    model = word2vec.Word2Vec(sentences = sentences, size = GrobalParament.train_size, window = GrobalParament.train_window, min_count = 20)
    model.save(model_out_put_path)
    print("训练完成")

if __name__ == "__main__":
    sentences = utils.preprocessing_text(GrobalParament.train_set_dir, GrobalParament.train_after_process_text_dir, GrobalParament.stop_word_dir)
    train(sentences, GrobalParament.model_output_path)
    model = word2vec.Word2Vec.load(GrobalParament.model_output_path)
    vocab = list(model.wv.vocab.keys())
    for e in model.most_similar(positive = ['地震'], topn = 10):
        print(e[0],e[1])
    print(len(vocab))