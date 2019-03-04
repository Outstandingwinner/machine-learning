


def get_word_ch(word):
    ch_lst=[]
    i = 0
    word_len = len(word)
    while i<word_len:
        ch_lst.append(word[i])
        i+=1
    return  ch_lst

ch_lst = []
status_lst = []
cur_ch_lst = get_word_ch(word="获取一个句话中每个字符对应的")
print(cur_ch_lst)
# print(cur_ch_lst[-1])
cur_ch_num = len(cur_ch_lst)
# print("长度:",cur_ch_num)
#
cur_status_lst = [0 for ch in range(cur_ch_num)]
# print(cur_status_lst)
