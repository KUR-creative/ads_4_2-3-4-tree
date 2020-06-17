from bisect import bisect
from collections import namedtuple


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

def tuple_insert(xs, idx, y):
    return xs[:idx] + (y,) + xs[idx:]

def insert(tree, key, max_n):
    node = tree
    
    idx = bisect(node.keys, key)
    new_keys = tuple_insert(node.keys, idx, key)
    
    if len(node.keys) < max_n:
        return node._replace(keys = new_keys)
    else:
        up_idx = 1 # (2-3, 2-3-4 just same as 1)
        up_key = node.keys[up_idx]
        return node._replace(
            is_leaf=False,
            keys=(up_key,),
            children=(leaf(*new_keys[:up_idx]),
                      leaf(*new_keys[up_idx + 1:])))

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
        

'''
print(rep(Node(False, (10, 12, 20),
                   (Node(True, (11,)), Node(True, (13,))))))
print(type(Node(False, (10, 12, 20),
                   (Node(True, (11,)), Node(True, (13,))))))
'''
