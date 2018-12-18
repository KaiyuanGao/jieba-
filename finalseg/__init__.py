from __future__ import absolute_import, unicode_literals
import re
import os
import sys
import pickle
from .._compat import *

MIN_FLOAT = -3.14e100 # 代表负无穷

PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"

# 每一个标记之前的标记是啥
PrevStatus = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}

Force_Split_Words = set([])
def load_model():
    # 得到马尔科夫链的开始概率，转移概率以及发射概率
    start_p = pickle.load(get_module_res("finalseg", PROB_START_P))
    trans_p = pickle.load(get_module_res("finalseg", PROB_TRANS_P))
    emit_p = pickle.load(get_module_res("finalseg", PROB_EMIT_P))
    return start_p, trans_p, emit_p

if sys.platform.startswith("java"):
    start_P, trans_P, emit_P = load_model()
else:
    from .prob_start import P as start_P
    from .prob_trans import P as trans_P
    from .prob_emit import P as emit_P


def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    viterbi函数会先计算各个初始状态的对数概率值，
    然后递推计算，每时刻某状态的对数概率值取决于
    上一时刻的对数概率值、
    上一时刻的状态到这一时刻的状态的转移概率、
    这一时刻状态转移到当前的字的发射概率三部分组成。
    """
    V = [{}]  # tabular表示Viterbi变量，下标表示时间
    path = {}  # 记录从当前状态回退路径
    # 时刻t=0，初始状态
    for y in states:  # init
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT) # 这里加号应该是乘号吧？
        path[y] = [y]
    # 时刻t = 1,...,len(obs) - 1
    for t in xrange(1, len(obs)):
        V.append({})
        newpath = {}
        # 当前时刻所处的各种可能的状态
        for y in states:
            # 获得发射概率对数
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            # 分别获得上一时刻的状态的概率对数、该状态到本时刻的状态的转移概率对数
            # 以及本时刻的状态的发射概率
            (prob, state) = max(
                [(V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0) for y0 in PrevStatus[y]])
            V[t][y] = prob
            # 将上一时刻最有的状态 + 这一时刻的状态
            newpath[y] = path[state] + [y]
        path = newpath
    # 最后一个时刻
    (prob, state) = max((V[len(obs) - 1][y], y) for y in 'ES')
    #返回最大概率对数和最有路径
    return (prob, path[state])


def __cut(sentence):
    """
    jieba首先会调用函数cut（sentence），cut函数会先将输入的句子进行解码，
    然后调用__cut()函数进行处理。该函数是实现HMM模型分词的主函数
    __cut()函数首先调用viterbi算法， 求出输入句子的隐藏状态，然后基于隐藏状态分词
    """
    global emit_P
    # 通过viterbi算法求出隐藏状态序列
    prob, pos_list = viterbi(sentence, 'BMES', start_P, trans_P, emit_P)
    begin, nexti = 0, 0
    # print pos_list, sentence
    # 基于隐藏状态进行分词
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        # 如果字所处的位置是开始位置 Begin
        if pos == 'B':
            begin = i
        # 如果字所处的位置是结束位置 END
        elif pos == 'E':
            # 这个子序列就一个分词
            yield sentence[begin:i + 1]
            nexti = i + 1
        # 如果单独成字 Single
        elif pos == 'S':
            yield char
            nexti = i + 1
    # 剩余的直接作为一个分词，返回
    if nexti < len(sentence):
        yield sentence[nexti:]

re_han = re.compile("([\u4E00-\u9FD5]+)")
re_skip = re.compile("([a-zA-Z0-9]+(?:\.\d+)?%?)")


def add_force_split(word):
    global Force_Split_Words
    Force_Split_Words.add(word)

def cut(sentence):
    sentence = strdecode(sentence)
    blocks = re_han.split(sentence)
    for blk in blocks:
        if re_han.match(blk):
            for word in __cut(blk):
                if word not in Force_Split_Words:
                    yield word
                else:
                    for c in word:
                        yield c
        else:
            tmp = re_skip.split(blk)
            for x in tmp:
                if x:
                    yield x
