from pprint import pprint
from bisect import bisect
from collections import namedtuple

import funcy as F


Node = namedtuple(
    'Node', 'is_leaf keys children', defaults=[()])

def leaf(*keys):
    return Node(True, tuple(keys), ())

def btree(max_n, *keys):
    root_key,*last = keys
    root = leaf(root_key)
    for key in last:
        root = insert(root, key, max_n)
    return root

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

def assert_valid(tree, max_n, input_keys):
    ks = all_keys(tree)
    ns = all_nodes(tree)
    #print('lm', len(input_keys), max_n)
    if len(input_keys) > max_n:
        assert (not tree.is_leaf), 'root is not leaf'
    assert len(input_keys) == len(ks), \
        f'{len(input_keys)} == {len(ks)}: number of keys inserted/flattend btree are not same'
    assert tuple(sorted(input_keys)) == ks, \
        f'{sorted(input_keys)} != {ks} keys from dfs are not sorted'
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
        
#--------------------------------------------------------------
def tuple_insert(tup, idx, val):
    return tup[:idx] + (val,) + tup[idx:]
def tuple_update(tup, idx, *val):
    return tup[:idx] + (*val,) + tup[idx+1:]
def tuple_omit(tup, idx):
    return tup[:idx] + tup[idx+1:]

up_idx = 1 # (2-3, 2-3-4 just same as 1)
def split_node(node, max_n):
    ''' a-b-c => b     a-b-c-d => b
                a c              a c-d '''
    #assert len(node.keys) > max_n, f'{len(node.keys)} <= {max_n}, \n{node}'
    global up_idx
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

# max_n = 2: 2-3
# max_n = 3: 2-3-4
def insert(node, key, max_n):
    idx = bisect(node.keys, key)
    if node.is_leaf:
        new_keys = tuple_insert(node.keys, idx, key)
        new_node = node._replace(keys = new_keys)
        return(new_node if len(new_keys) <= max_n # just insert to leaf
          else split_node(new_node, max_n)) # split
    else:
        old_child = node.children[idx]
        child = insert(node.children[idx], key, max_n)
        has_split_child = (
            len(old_child.keys) > len(child.keys))

        if not has_split_child: # key is just inserted
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
            return(merged if len(merged.keys) <= max_n # just insert to leaf
              else split_node(merged, max_n)) # split
        
def update(node, idxes, new_node):
    print('----')
    print(tuple(node))
    if not idxes: # empty
        return new_node
    else:
        idx, *last = idxes
        return node._replace(
            children = tuple_update(
                node.children,
                idx,
                update(node.children[idx], last, new_node)))
    
tree = btree(2, 2,5,7,8)
tree = btree(2, 100, 50, 150, 200, 120, 135, 140, 210, 220)
print(tuple(tree))
print(tuple(update(tree, [1], leaf(-1))))
print('====')
print(tuple(update(tree, [1,2], leaf(-1))))

exit()
    
def delete(tree, key, max_n):
    # get path root to leaf
    node = tree
    nodes = [node]
    idxes = []
    while not node.is_leaf:
        node_idx = bisect(node.keys, key)
        next_node = node.children[node_idx]
        nodes.append(next_node)
        idxes.append(node_idx)
        node = next_node

    leaf = nodes[-1]
    idx = bisect(leaf.keys, key)
    new_leaf = leaf._replace(
        keys=tuple_omit(leaf.keys, idx)
    )
    return update(root, idxes, new_leaf)
    #pprint(nodes)
    print('idxes:', idxes)
    return tree
#--------------------------------------------------------------

tree = btree(2, 2,5,7,8)
print('-------- before --------')
pprint(tuple(tree))
print('-------- after --------')
pprint(tuple(delete(tree, 7, 2)))
'''
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
max_n = 2
#max_n = 3
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
    assert_valid(tree, max_n, keys[:end])
'''
