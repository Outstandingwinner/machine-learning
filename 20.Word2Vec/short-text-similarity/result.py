#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce

import GrobalParament
import utils


#构建文档字典
def bulid_dict(file_path):

    doc_dict = dict()
    with open(file_path, "r", encoding=GrobalParament.encoding) as f_reader:
        for line in f_reader:
            line = utils.delete_r_n(line)
            line_list = line.split(",")
            if len(line_list) == 2:
                doc_dict[line_list[0]] = line_list[1]

    return doc_dict

def sim_result_out(sim_out_path, test_dict, train_dict, result_path):
    f_writer = open(result_path, "w", encoding=GrobalParament.encoding)

    with open(sim_out_path, "r", encoding=GrobalParament.encoding) as f_reader:
        for line in f_reader:
            line = utils.delete_r_n(line)
            line_list = line.split(",")
            if len(line_list) == 2:
                test_docId = line_list[0]
                sim_result = test_docId + "," + test_dict[test_docId] + "\n" + "***最相似的前20个***\n"

                train_docId_list = line_list[1].split()
                for id in train_docId_list:
                    sim_result = sim_result + id + "," + train_dict[id] + "\n"

                f_writer.write(sim_result)
                f_writer.write("*********************************************\n")
                f_writer.flush()

    f_writer.close()




if __name__ == "__main__":
    train_dict = bulid_dict(GrobalParament.train_set_dir)
    test_dict = bulid_dict(GrobalParament.test_set_dir)

    sim_result_out(GrobalParament.result_out_path, test_dict, train_dict, GrobalParament.sim_result_path)
