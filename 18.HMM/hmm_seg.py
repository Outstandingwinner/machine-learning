
test_txt = '.\\data\\test.txt'
mod_path = '.\\data\\hmm_mod.data'
def get_word_ch(word):
    ch_lst=[]
    i = 0
    word_len = len(word)
    while i<word_len:
        ch_lst.append(word[i])
        i+=1
    return  ch_lst

# init 参数
STATUS_NUM = 4

B = [dict() for col in range(STATUS_NUM)]

f_mod =open(mod_path,'r',encoding='utf-8')

# 加载初始log概率
pi = [float(i) for i in f_mod.readline().split()]

# 加载转移矩阵
A = [[float(j) for j in f_mod.readline().split()] for i in range(STATUS_NUM)]

# 加载发射矩阵
for i in range(STATUS_NUM):
    B_i_tokens = f_mod.readline().split()
    token_num = len(B_i_tokens)
    j = 0
    while j+1 < token_num:
        B[i][B_i_tokens[j]] = float(B_i_tokens[j+1])
        # if j<= 10:
        #     print(str(B_i_tokens[j])+":"+B_i_tokens[j+1])
        j += 2
f_mod.close()

# seg切词

f_txt = open(test_txt,'r',encoding='utf-8')
while True:
    line = f_txt.readline()
    if not line: break
    line = line.strip()

    ch_lst = get_word_ch(line)
    ch_num = len(ch_lst)

    if ch_num < 1:
        print()
        continue

    # 初始化状态矩阵
    status_matrix = [[[0.0, 0] for col in range(ch_num)] for st in range(STATUS_NUM)]

    # init
    for i in range(STATUS_NUM):
        if ch_lst[0] in B[i]:
            cur_B = B[i][ch_lst[0]]
        else:cur_B = -100000.0
        if pi[i] == 0.0:
            cur_pi = -100000.0
        else:
            cur_pi = pi[i]
        status_matrix[i][0][0] = cur_pi+cur_B
        status_matrix[i][0][1] = i

    # viterbi算法
    for i in range(1,ch_num):
        for j in range(STATUS_NUM):
            max_p = None
            max_status = None
            for k in range(STATUS_NUM):
                cur_A = A[k][j]
                if cur_A == 0.0: cur_A = -10000.0
                cur_p = status_matrix[k][i-1][0] +cur_A
                if max_p is None or max_p<cur_p:
                    max_p=cur_p
                    max_status = k
            if ch_lst[i] in B[j]:
                cur_B = B[j][ch_lst[i]]
            else:
                cur_B = -100000.0
            status_matrix[j][i][0] = max_p +cur_B
            status_matrix[j][i][1] = max_status

    # 获取最终最大的概率，对应找到这条路径
    max_end_p = None
    max_end_status = None
    for i in range(STATUS_NUM):
        if max_end_p is None or status_matrix[i][ch_num-1][0]>max_end_p:
            max_end_p = status_matrix[i][ch_num-1][0]
            max_end_status = i
    best_status_lst = [0 for ch in range(ch_num)]
    best_status_lst[ch_num-1] = max_end_status

    i = ch_num -1
    cur_best_status = max_end_status

    while i>0:
        print("cur_best_status=", cur_best_status)
        pre_best_status = status_matrix[cur_best_status][i][1]
        print("pre_best_status=",pre_best_status)
        # 存最有路劲
        best_status_lst[i-1] = pre_best_status
        cur_best_status = pre_best_status
        i -= 1
    print(line)
    print(best_status_lst)

    out_s = ""
    out_s+= ch_lst[0]
    for i in range(1,ch_num):
        if best_status_lst[i-1] in {2,3} or best_status_lst[i] in {0,3}:
            out_s += "  "
        out_s+=str(ch_lst[i])
    out_s+='\n'
    print(out_s)



count = 0
for i in range(1,ch_num):
    for j in range(STATUS_NUM):
       count +=1
       print(status_matrix[j][i][1], end=" ")
       if count % 11 == 0:
           print("")


