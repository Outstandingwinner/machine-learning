
import sys

file1 = sys.argv[1]
file2 = sys.argv[2]

t_list = []

with open(file1,'r') as fd:
    for line in fd:
        t_label = line.strip().split(' ')[0]
        t_list.append(t_label)

p_list = []
with open(file2,'r') as fd:
    for line in fd:
        p_label = line.strip()
        p_list.append(p_label)

if len(t_list) == len(p_list):
    for i in range(0,len(t_list)):
        print '\t'.join([ t_list[i], p_list[i] ])

