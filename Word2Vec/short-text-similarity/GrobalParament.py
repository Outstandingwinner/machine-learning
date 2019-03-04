#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce

encoding = "utf-8" #编码设置

stop_word_dir = "data/stop_words.txt"

train_set_dir = "data/train.csv"

train_after_process_text_dir = "data/train_after_process.csv"

test_set_dir = "data/test.csv"

test_after_process_text_dir = "data/test_after_process.csv"


model_output_path = "model\word2vec_news.model"

train_num = 485686

test_num = 50

result_out_path = "data/sim_result.csv"

sim_result_path = "data/result.txt"

#word2vec参数
train_size = 150
train_window = 5

