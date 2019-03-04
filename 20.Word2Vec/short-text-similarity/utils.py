#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce

import GrobalParament
import jieba
import numpy as np
import pandas as pd


#去掉回车换行
def delete_r_n(line):
    return line.replace("\r","").replace("\n","").strip()

#读取停用词
def get_stop_words(stop_words_dir):
    stop_words = []

    with open(stop_words_dir, "r", encoding=GrobalParament.encoding) as f_reader:
        for line in f_reader:
            line = delete_r_n(line)
            stop_words.append(line)

    stop_words = set(stop_words)
    return stop_words


#结巴精准分词
def jieba_cut(content, stop_words):
    word_list = []

    if content != "" and content is not None:
        seg_list = jieba.cut(content)
        for word in seg_list:
            if word not in stop_words:
                word_list.append(word)

    return word_list

#结巴索索引擎分词
def jieba_cut_for_search(content, stop_words):
    word_list = []

    if content != "" and content is not None:
        seg_list = jieba.cut_for_search(content)
        for word in seg_list:
            if word not in stop_words:
                word_list.append(word)

    return word_list

#清除不在词汇表中的词语
def clear_word_from_vocab(word_list, vocab):
    new_word_list = []

    for word in word_list:
        if word in vocab:
            new_word_list.append(word)

    return new_word_list

#文本预处理
def preprocessing_text_pd(text_dir, after_process_text_dir, stop_words_dir):
    stop_words = get_stop_words(stop_words_dir)
    setences = []
    df = pd.read_csv(text_dir)
    for index, row in df.iterrows():
        print(index)
        title = delete_r_n(row['title'])
        word_list = jieba_cut(title, stop_words)
        df.loc[index,'title'] = " ".join(word_list)
        setences.append(word_list)
    df.to_csv(after_process_text_dir,encoding=GrobalParament.encoding, index=False)
    return setences

#文本预处理第二种方式
def preprocessing_text(text_dir, after_process_text_dir, stop_words_dir):
    stop_words = get_stop_words(stop_words_dir)
    setences = []
    f_writer = open(after_process_text_dir,"w", encoding=GrobalParament.encoding)
    #count = 0
    with open(text_dir, "r", encoding=GrobalParament.encoding) as f_reader:
        for line in f_reader:
            line_list = line.split(",")
            if len(line_list) == 2:
                line_list[1] = delete_r_n(line_list[1])
                word_list = jieba_cut(line_list[1], stop_words)
                setences.append(word_list)
                f_writer.write(line_list[0] + "," + " ".join(word_list) + "\n")
                f_writer.flush()
                #count = count + 1
                #print(count)
            else:
                print(line)

    f_writer.close()
    return setences





if __name__ == "__main__":
    stop_words = get_stop_words(GrobalParament.stop_word_dir)
    content = "我毕业于北京理工大学，现就职于中国科学院计算技术研究所。"
    word_list = jieba_cut(content,stop_words)
    print("word_list=",word_list)
    word_list = jieba_cut_for_search(content,stop_words)
    print("word_list=", word_list)
    setences = preprocessing_text(GrobalParament.test_set_dir, GrobalParament.test_after_process_text_dir, GrobalParament.stop_word_dir)
    #setences = preprocessing_text_pd(GrobalParament.train_set_dir, GrobalParament.train_after_process_text_dir, GrobalParament.stop_word_dir)
    print(setences[:10])
