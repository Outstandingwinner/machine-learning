#!/usr/local/bin/python

import sys

cur_ii_pair = None
score = 0.0

for line in sys.stdin:
	ii_pair, s = line.strip().split("\t")
	if not cur_ii_pair:
		cur_ii_pair = ii_pair
	if ii_pair != cur_ii_pair:
		item_a, item_b = cur_ii_pair.split('')
		print "%s\t%s\t%s" % (item_a, item_b, score)
		cur_ii_pair = ii_pair
		score = 0.0

	score += float(s)

item_a, item_b = cur_ii_pair.split('')
print "%s\t%s\t%s" % (item_a, item_b, score)
