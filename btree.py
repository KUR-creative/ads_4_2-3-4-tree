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
def tuple_update(tup, idx, *val):
    return tup[:idx] + (*val,) + tup[idx+1:]
def tuple_omit(tup, idx):
    return tup[:idx] + tup[idx+1:]

'''
def index(arr, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(arr, x)
    if i != len(arr) and arr[i] == x:
        return i
    # if x not in arr, return None
'''

up_idx = 1 # (2-3, 2-3-4 just same as 1)
#def split_node(node):
def split_node(node, max_n):
    ''' a-b-c => b     a-b-c-d => b
                a c              a c-d '''
    assert len(node.keys) > max_n, f'{len(node.keys)} <= {max_n}, \n{node}'
    #print('split!')
    return node._replace(
        is_leaf = False,
        keys = (node.keys[up_idx],),
        children = (
            Node(is_leaf = node.is_leaf,
                 keys = node.keys[:up_idx],
                 children = node.children[:up_idx+1]),
            Node(is_leaf = node.is_leaf,
                 keys = node.keys[up_idx + 1:],
                 children = node.children[up_idx+1:])))

def insert(node, key, max_n):
    #print('node:', node)
    idx = bisect(node.keys, key)
    if node.is_leaf:
        new_keys = tuple_insert(node.keys, idx, key)
        new_node = node._replace(keys = new_keys)
        return(new_node if len(new_keys) <= max_n # just insert to leaf
          else split_node(new_node, max_n)) # split
    else:
        #print('-------------------')
        old_child = node.children[idx]
        #print('before c', tuple(old_child))
        child = insert(node.children[idx], key, max_n)
        #print(':', node)
        #print(':', child)
        #print('n',tuple(node))
        
        has_split_child = (
            len(old_child.keys) > len(child.keys))
        #no_split_child = (node.children[idx].children == child.children)

        #print(' after c', tuple(child))
        #print('has_split_child:', has_split_child)
        #print('-------------------')
        if not has_split_child: # key is just inserted to leaf
            return node._replace(
                children = tuple_update(
                    node.children, idx, child))
        else: # child has been split.
            excerpt = child
            up_key = excerpt.keys[0]
            merged = node._replace(
                keys = tuple_insert(
                    node.keys, idx, up_key),
                children = tuple_update(
                    node.children, idx, *excerpt.children))
            #print('merged!')
            #print('exc', excerpt)
            #print('is_split', len(merged.keys) <= max_n)
            return(merged if len(merged.keys) <= max_n # just insert to leaf
              else split_node(merged, max_n)) # split
            '''
            print('idx', idx)
            print('n',tuple(node))
            print('c',tuple(child))
            print('m',tuple(merged))
            '''

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
keys = (100, 50, 150, 200, 120)
keys = (100, 50, 150, 200, 120, 135)
keys = (100, 50, 150, 200, 120, 135, 140)
keys = (100, 50, 150, 200, 120, 135, 140, 170)
keys = (100, 50, 150, 200, 120, 135, 140, 170, 250)
keys = (100, 50, 150, 200, 120, 135, 140, 170, 250, 145)
keys = (100, 50, 150, 200, 120, 135, 140, 170, 250, 145, 142)
keys = (100,50,150,200,120,135,140,170,250,145,142,-10,-25, 130)
keys = (100,50,150,200,120,135,140,170,250,145,142,-10,-25, 130, 140, 134)
from pprint import pprint
#max_n = 2
max_n = 3
tree = Node(True, keys[:1])
print('-------------------======')
pprint(tuple(tree))
for end,key in enumerate(keys[1:], start=2):
    print('---- inp:', key, '----')
    tree = insert(tree, key, max_n)
    #pprint(tuple(tree))
    pprint(tuple(tree))

    ks = all_keys(tree)
    ns = all_nodes(tree)
    # pprint(tree)
    if len(keys[:end]) > max_n:
        assert (not tree.is_leaf), 'root is not leaf'
    assert len(keys[:end]) == len(ks), \
        f'{len(keys[:end])} == {len(ks)}: number of keys inserted/flattend btree are not same'
    assert tuple(sorted(keys[:end])) == ks, 'keys from dfs are sorted'
    assert all((not is_invalid(n, max_n)) for n in ns), \
        'all nodes are valid'
    assert all(len(n.children) == 0 for n in ns if n.is_leaf), \
        'all leaves have no children'
    assert all(map(
        lambda node: all([
            n1.is_leaf == n2.is_leaf
            for n1,n2 in F.pairwise(node.children)]),
        ns
    )),'leaves are all same h (children are all leaves or not) '
'''
print('result:')
pprint(dfs(tree))

nodes = all_nodes(tree)
print('---- check ----')
for node in nodes:
    print(node.children)
    print()
'''
#pprint(tuple(tree))

