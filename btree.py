from pprint import pprint
from bisect import bisect, bisect_left
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
def is_empty(coll):
    return (not coll)

def tuple_insert(tup, idx, val):
    return tup[:idx] + (val,) + tup[idx:]
def tup_update(tup, idx, *val):
    return tup[:idx] + (*val,) + tup[idx+1:]
def tup_omit(tup, idx): # TODO: negative idx case
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
                children = tup_update(
                    node.children, idx, child))
        else: # child has been split.
            excerpt = child
            up_key = excerpt.keys[0]
            merged = node._replace(
                keys = tuple_insert(
                    node.keys, idx, up_key),
                children = tup_update(
                    node.children, idx, *excerpt.children))
            return(merged if len(merged.keys) <= max_n # just insert to leaf
              else split_node(merged, max_n)) # split
        
def update(node, idxes, new_node):
    #print('----')
    #print(tuple(node))
    if not idxes: # empty
        return new_node
    else:
        idx, *last = idxes
        return node._replace(
            children = tup_update(
                node.children,
                idx,
                update(node.children[idx], last, new_node)))
'''
tree = btree(2, 2,5,7,8)
tree = btree(2, 100, 50, 150, 200, 120, 135, 140, 210, 220)
print(tuple(tree))
print(tuple(update(tree, [1], leaf(-1))))
print('====')
print(tuple(update(tree, [1,2], leaf(-1))))
exit()
'''
def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return None

def sibling_idxes(children, idx):
    ''' return: index(es) of sibling, if no sibling, []. '''
    idx = idx + len(children) if idx < 0 else idx
    left = idx - 1 if idx > 0 else 0
    right= idx + 1
    idxes = list(range(len(children)))
    return idxes[left:idx] + idxes[right:right+1]

def theft_victim(children, sibling_idxes, target_idx):
    ''' return: valid sibling, sib idx, key idx. left first. '''
    for sib_idx in sibling_idxes:
        victim = children[sib_idx]
        len_keys = len(victim.keys)
        if len_keys > 1:
            return (victim, sib_idx,
                    len_keys - 1 if sib_idx == target_idx - 1
                    else 0)
    return None, None, None
    
def get_path(tree, key): # Get path root to leaf
    node = tree
    nodes = [node]
    idxes = []
    while not node.is_leaf:
        node_idx = bisect(node.keys, key)
        next_node = node.children[node_idx]
        nodes.append(next_node)
        idxes.append(node_idx)
        node = next_node
    return nodes, idxes

def steal(tree, idxes, empty_node,
          parent, parent_idx, parent_key_idx,
          victim, victim_idx, victim_key_idx):
    # victim is not None case...
    parent_key = parent.keys[parent_key_idx] 
    stolen_key = victim.keys[victim_key_idx]
    # make updated nodes
    new_target = empty_node._replace(keys = (parent_key,))
    new_parent = parent._replace(
        keys = tup_update(
            parent.keys, parent_key_idx, stolen_key))
    new_victim = victim._replace(
        keys = tup_omit(victim.keys, victim_key_idx))
    # Update parent, sibling, target. TODO: add `updates`
    new_tree = update(tree, idxes[:-1], new_parent)#parent
    new_tree = update(new_tree, idxes[:-1] + [victim_idx], new_victim) #sibling
    new_tree = update(new_tree, idxes, new_target) #target
    return new_tree

def merge(tree, idxes, leaf, leaf_idx, parent, sib_idxes):
    sib_idx = sib_idxes[0]
    sibling = parent.children[sib_idx]

    assert leaf_idx >= 0
    parent_key_idx =(
        leaf_idx - 1 if leaf_idx > 0 else leaf_idx)
    #   merge with left,          merge with right

    new_keys = (
        sibling.keys[0], parent.keys[parent_key_idx]
    ) if leaf_idx > 0 else (
        parent.keys[parent_key_idx], sibling.keys[0]
    )

    merged = leaf._replace(keys = new_keys)
    merged_idx = parent_key_idx # idx calc
    omitted_parent = parent._replace(
        keys = tup_omit(parent.keys, parent_key_idx),
        children = tup_omit(parent.children, leaf_idx))
    new_parent = omitted_parent._replace(
        children = tup_update(
            omitted_parent.children,
            merged_idx,
            merged))
    return update(tree, idxes[:-1], new_parent)

def delete(tree, key, max_n):
    # Get path root to leaf
    nodes, idxes = get_path(tree, key)
    # Update leaf
    leaf = nodes[-1]
    leaf_idx = idxes[-1] # idx in idxes
    leaf_key_idx = index(leaf.keys, key) # idx in leaf.keys
    new_leaf = leaf._replace(
        keys =(leaf.keys if leaf_key_idx is None else
               tup_omit(leaf.keys, leaf_key_idx))
    )
    # ---- Make valid b-tree, replace, steal, merge ----
    if is_empty(new_leaf.keys):
        parent = nodes[-2] # TODO: not only -2
        parent_idx = None # TODO: need?
        
        sib_idxes = sibling_idxes(parent.children, leaf_idx)
        victim, victim_idx, victim_key_idx = theft_victim(
            parent.children, sib_idxes, leaf_idx)
        
        assert sib_idxes, 'It cannot be empty'
        if victim is None:
            return merge(
                tree, idxes, leaf, leaf_idx, parent, sib_idxes)
        else:
            parent_key_idx =(
                leaf_idx if leaf_idx + 1 == victim_idx else
                leaf_idx - 1 if leaf_idx - 1 == victim_idx else
                None) # it crashes!
            return steal(
                tree, idxes, new_leaf,
                parent, parent_idx, parent_key_idx,
                victim, victim_idx, victim_key_idx)
        
    return update(tree, idxes, new_leaf)

'''
# victim is not None case...
parent_key_idx = 0 # TODO: not 0 case
parent_key = parent.keys[parent_key_idx] 
stolen_key = victim.keys[victim_key_idx]

new_target = new_leaf._replace(keys = (parent_key,))
new_parent = parent._replace(
    keys = tup_update(parent.keys, 0, stolen_key))
new_victim = victim._replace(
    keys = tup_omit(victim.keys, victim_key_idx))
# Update parent, sibling, target. TODO: add `updates`
new_tree = update(
    tree, idxes[:-1], new_parent)#parent
new_tree = update(
    new_tree, idxes[:-1] + [victim_idx], new_victim) #sibling
new_tree = update(
    new_tree, idxes, new_target) #target
return new_tree
'''
#--------------------------------------------------------------

tree = btree(2, 2,5,7,8)
tree = btree(2, 5,7,8,6)
max_n = 3; tree = btree(max_n, 3,5,7,9,11,13,15,17)
max_n = 2; tree = btree(max_n, 4,5,7,8,10)
max_n = 2; tree = btree(max_n, 5,10,15,20,25,30)
print('-------- before --------')
pprint(tuple(tree))
print('-------- after --------')
#pprint(tuple(delete(tree, 7, max_n)))
pprint(tuple(delete(tree, 5, max_n)))
pprint(tuple(delete(delete(tree, 5, max_n), 25, max_n)))
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
