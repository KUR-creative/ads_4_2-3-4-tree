from bisect import bisect
from collections import namedtuple

import funcy as F


Node = namedtuple(
    'Node', 'is_leaf keys children', defaults=[()])

def rep(node):
    return '[{}{}|{}|{}]'.format(
        'L' if node.is_leaf else 'M',
        ' '.join(map(str, node.keys)),
        ' '.join(str(c.keys[0]) + '.' for c in node.children)
    ) if type(node) is Node else 'NotNode'

# max_n = 2: 2-3
# max_n = 3: 2-3-4
def leaf(*keys):
    return Node(True, tuple(keys), ())

def is_invalid(node, max_n, min_n=1):
    if type(node) is not Node:
        return 'NotNode'
    n = len(node.keys)
    keys = node.keys
    n_children = len(node.children)
    ret = ''.join([
        f'n = {n} > {max_n} = max_n \n' if n > max_n else '',
        f'n = {n} < {min_n} = min_n \n' if n < min_n else '',
        
        f'#children = {n_children} != {n + 1} = n + 1 \n'
        if (not node.is_leaf) and n_children != n + 1 else '',
        
        f'invalid leaf - children: {node.children}'
        if node.is_leaf and len(node.children) > 0 else '',
        f'invalid non-leaf: no children: {node.children}'
        if (not node.is_leaf) and len(node.children) == 0 else '',
        
        f'keys not sorted: {keys} \n' if any(
            keys[i] > keys[i+1] for i in range(len(keys)-1)
        ) else ''
    ])
    return ret # If valid, return '' (falsey value)

def dfs(node):
    print(node.keys)
    for child in node.children:
        dfs(child)
        
def intersect_seq(children, keys):
    return F.butlast(F.cat(zip(
        children, keys + type(keys)([None])
    )))
        
def all_keys(root):
    if root.is_leaf:
        return root.keys
    else:
        keys = ()
        for x in intersect_seq(root.children, root.keys):
            keys += ((x,) if type(x) is int else all_keys(x))
        return keys

def all_nodes(root, nodes=()):
    nodes = (root,)
    for node in root.children:
        nodes += all_nodes(node)
    return nodes
        
def tuple_insert(tup, idx, val):
    return tup[:idx] + (val,) + tup[idx:]
def tuple_update(tup, idx, val):
    ret = list(tup)
    ret[idx] = val
    return tuple(ret)

'''
def index(arr, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(arr, x)
    if i != len(arr) and arr[i] == x:
        return i
    # if x not in arr, return None
'''

up_idx = 1 # (2-3, 2-3-4 just same as 1)
def insert(node, key, max_n):
    idx = bisect(node.keys, key)
    if node.is_leaf:
        new_keys = tuple_insert(node.keys, idx, key)
        if len(node.keys) < max_n: # just insert to leaf
            return node._replace(keys = new_keys)
        else: # insert to leaf and return splitted.
            up_key = new_keys[up_idx]
            return node._replace(
                is_leaf = False,
                keys = (up_key,),
                children = (leaf(*new_keys[:up_idx]),
                            leaf(*new_keys[up_idx + 1:])))
    else:
        child = insert(node.children[idx], key, max_n)
        #print(':', node)
        #print(':', child)
        if child.is_leaf: # key is just inserted to leaf
            return node._replace(
                children = tuple_update(
                    node.children, idx, child))
        else: # leaf is splitted.
            pass

def split_child(unfull_parent, child_idx, max_n, min_n=1):
    ''' 
    Because it support 2-3 tree, it is different to CLRS. 
    Add to child, and split. that's it.
    '''
    child = unfull_parent.children[child_idx] # child is full.
    
    up_idx = max_n // 2 # index of elem to go parent 
    up_key = child.keys[up_idx]
    
    #l_child = Node(
        #child.is_leaf, min_n, max_n, child.keys[:up_idx],
        
def dfs(root):
    #tups = (tuple(root), )
    tups = (tuple(root), )
    for node in root.children:
        tups += dfs(node)
    return tups

keys = (100, 50)
keys = (100, 50, 150)
keys = (100, 50, 150, 200)
from pprint import pprint
max_n = 2
tree = Node(True, keys[:1])
print('-------------------======')
pprint(tuple(tree))
for key in keys[1:]:
    print('---- inp:', key, '----')
    tree = insert(tree, key, max_n)
    pprint(tuple(tree))
print('result:')
#pprint(tuple(tree))
pprint(dfs(tree))
