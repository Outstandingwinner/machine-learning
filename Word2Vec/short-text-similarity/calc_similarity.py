#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce
#微信号 wibruce

import GrobalParament
import utils


from gensim.models.word2vec import Word2Vec

def calc_sim(model_path, train_path, test_path, out_result_path):
    model = Word2Vec.load(model_path)
    vocab = set(model.wv.vocab.keys())

    #计算相似度
    f_writer = open(out_result_path,"w", encoding=GrobalParament.encoding)
    with open(test_path, "r", encoding=GrobalParament.encoding) as f_test_reader:
        f_test_reader.readline()
        for test_line in f_test_reader:
            test_line = utils.delete_r_n(test_line)
            test_line_list = test_line.split(",")
            print("测试集合：" + test_line_list[0])

            test_word_list = test_line_list[1].split()
            test_word_list = utils.clear_word_from_vocab(test_word_list, vocab)
            sim_score = dict()

            with open(train_path, "r" ,encoding = GrobalParament.encoding) as f_train_reader:
                f_train_reader.readline()
                for train_line in f_train_reader:
                    train_line = utils.delete_r_n(train_line)
                    train_line_list = train_line.split(",")
                    if len(train_line_list) == 2:
                        print("训练集合：" + train_line_list[0])
                        train_word_list = train_line_list[1].split()
                        train_word_list = utils.clear_word_from_vocab(train_word_list,vocab)

                        if len(train_word_list) > 0:
                            sim_score[train_line_list[0]] = model.n_similarity(test_word_list, train_word_list)


            sim_score = sorted(sim_score.items(), key=lambda d:d[1], reverse=True)
            print("开始计算前20个最相似的")
            train_doc_num = ""
            for items in sim_score[:20]:
                train_doc_num = train_doc_num + items[0] + " "
            f_writer.write(test_line_list[0] + "," + train_doc_num.strip() + "\n")
            f_writer.flush()
    f_writer.close()






if __name__ == "__main__":
    calc_sim(GrobalParament.model_output_path, GrobalParament.train_after_process_text_dir, GrobalParament.test_after_process_text_dir, GrobalParament.result_out_path)