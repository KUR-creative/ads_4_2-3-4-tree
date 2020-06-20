from time import time
from random import randint, shuffle

import funcy as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from btree import btree, insert, delete, find


def shuffled(lst):
    ''' return copied, shuffled list '''
    ret = lst[:]
    shuffle(ret)
    return ret

def run_time(fn):
    s = time()
    ret = fn()
    t = time()
    return t - s, ret

Ns = list(range(1000, 100000, 1000))
#Ns = list(range(100, 1000, 100))
times = dict(insert = {23: [], 234: []},
             find   = {23: [], 234: []},
             delete = {23: [], 234: []})
for N in tqdm(Ns):
    keys = shuffled(list(range(N)))
    
    inserts, _ = F.lsplit_at(N // 2, keys)
    finds = shuffled(keys)[:N//2] # different order to inserts
    deletes = shuffled(keys)[:N//2] # different order to inserts

    t23ins_time, t23 = run_time(lambda: btree(2, *inserts))
    times['insert'][23].append(t23ins_time)
    t234ins_time, t234 = run_time(lambda: btree(3, *inserts))
    times['insert'][234].append(t234ins_time)

    t23find_time = 0
    for key in finds:
        t23find_time += run_time(lambda: find(t23, key))[0]
    times['find'][23].append(t23find_time)
    t234find_time = 0
    for key in finds:
        t234find_time += run_time(lambda: find(t234, key))[0]
    times['find'][234].append(t234find_time)

    t23del_time = 0
    for key in deletes:
        if t23:
            a_t23del_time, t23 = run_time(
                lambda: delete(t23, key))
            t23del_time += a_t23del_time
    times['delete'][23].append(t23del_time)
    t234del_time = 0
    for key in deletes:
        if t234:
            a_t234del_time, t234 = run_time(
                lambda: delete(t234, key))
            t234del_time += a_t234del_time
    times['delete'][234].append(t234del_time)

from pprint import pprint
'''
pprint(times['insert'])
pprint(times['find'])
pprint(times['delete'])
'''
pprint(times)

for what in ['insert','find','delete']:
    plt.plot(Ns, times[what][23],  label=f'2-3 tree {what}')
    plt.plot(Ns, times[what][234], label=f'2-3-4 tree {what}')
    plt.xlabel('number of items')
    plt.ylabel('time (sec)')
    plt.legend()
    plt.show()

