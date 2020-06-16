from collections import namedtuple


Node = namedtuple(
    'Node',
    'is_leaf min_n max_n keys children', defaults=[(), ()])

def rep(node):
    return '[{}{}|{}|{}]'.format(
        node.max_n, 'L' if node.is_leaf else 'M',
        ' '.join(map(str, node.keys)),
        ' '.join(str(c.keys[0]) + '.' for c in node.children)
    ) if type(node) is Node else 'NotNode'

def Node2(is_leaf, keys, children=()): # 2 = max num of keys
    return Node(is_leaf, 1, 2, tuple(keys), children) # 2-3
def Node3(is_leaf, keys, children=()): # 3 = max num of keys
    return Node(is_leaf, 1, 3, tuple(keys), children) # 2-3-4

def is_invalid(node):
    if type(node) is not Node:
        return 'NotNode'
    n = len(node.keys)
    keys = node.keys
    max_n = node.max_n
    min_n = node.min_n
    n_children = len(node.children)
    ret = ''.join([
        f'n = {n} > {max_n} = max_n \n' if n > max_n else '',
        f'n = {n} < {min_n} = min_n \n' if n < min_n else '',
        f'#children = {n_children} != {n + 1} = n + 1 \n'
        if (not node.is_leaf) and n_children != n + 1 else '',
        f'keys not sorted: {keys} \n' if any(
            keys[i] > keys[i+1] for i in range(len(keys)-1)
        ) else ''
    ])
    return ret # If valid, return '' (falsey value)

def split_child(unfull_parent, child_idx):
    ''' 
    Because it support 2-3 tree, it is different to CLRS. 
    Add to child, and split. that's it.
    '''
    child = unfull_parent.children[child_idx] # child is full.
    min_n, max_n = child.min_n, child.max_n
    
    up_idx = max_n // 2 # index of elem to go parent 
    up_key = child.keys[up_idx]
    
    l_child = Node(
        child.is_leaf, min_n, max_n,
        child.keys[:up_idx],
        

print(rep(Node3(False, (10, 12, 20),
                   (Node3(True, (11,)), Node3(True, (13,))))))
print(type(Node3(False, (10, 12, 20),
                   (Node3(True, (11,)), Node3(True, (13,))))))
