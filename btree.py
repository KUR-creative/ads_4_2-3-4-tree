import random
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
    if node is None:
        return None
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
    if tree is None:
        return None
    ks = all_keys(tree)
    ns = all_nodes(tree)
    #print('lm', len(input_keys), max_n)
    if len(input_keys) > max_n:
        assert (not tree.is_leaf), 'root is not leaf'
    assert len(input_keys) == len(ks), \
        f'len({input_keys}) != len({ks}): number of keys inserted/flattend btree are not same'
    assert tuple(sorted(input_keys)) == ks, \
        f'{sorted(input_keys)} != {ks} keys from dfs are not sorted'
    for n in ns:
        assert not is_invalid(n, max_n), \
        f'some nodes are invalid:\n {is_invalid(n, max_n)}'
    assert all(len(n.children) == 0 for n in ns if n.is_leaf), \
        'some leaves have children'
    assert all(len(n.children) != 0 for n in ns if not n.is_leaf), \
        'some internal leaves have no children'
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

def tup_insert(tup, idx, *val):
    return tup[:idx] + (*val,) + tup[idx:]
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
    if node is None:
        return leaf(key)
    idx = bisect(node.keys, key)
    if node.is_leaf:
        new_keys = tup_insert(node.keys, idx, key)
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
                keys = tup_insert(
                    node.keys, idx, up_key),
                children = tup_update(
                    node.children, idx, *excerpt.children))
            return(merged if len(merged.keys) <= max_n # just insert to leaf
              else split_node(merged, max_n)) # split
        
def update(node, idxes, new_node):
    ''' 
    Go to deep along to idxes..
    if idxes are empty, then change the node with new node
    '''
    #print('--x--')
    #print(tuple(node))
    #print(':', idxes)
    #print(tuple(new_node))
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
    if idx is None:
        return []
    idx = idx + len(children) if idx < 0 else idx
    left = idx - 1 if idx > 0 else 0
    right = idx + 1
    idxes = list(range(len(children)))
    return idxes[left:idx] + idxes[right:right+1]

def theft_victim(children, sibling_idxes, target_idx):
    ''' return: omitted victim, victim idx, value to up. left first. '''
    for idx in sibling_idxes:
        victim = children[idx]
        keys = victim.keys
        
        len_keys = len(keys)
        key_idx = len_keys - 1 if idx == target_idx - 1 else 0
        up_key = keys[key_idx]
        if len_keys > 1:
            return (
                victim._replace(
                    keys=tup_omit(keys, key_idx)
                ),  # TODO: omit child too
                idx,
                up_key
            )
    return None, None, None
    
def get_path(tree, key): # Get path root to leaf
    '''
    returns:
    founds: t/f ---------- t/f ------------- ...
    nodes: node0 -------- node1 ------- ...
    idxes:         idx1 -------- idx2 ---- ...
    '''
    node = tree
    
    nodes = [node]
    idxes = []
    founds= [key in node.keys]
    while not node.is_leaf:
        node_idx = bisect(node.keys, key)
        print('n keys', node.keys, 'key', key, 'idx', node_idx)

        next_node = node.children[node_idx]
        nodes.append(next_node)
        idxes.append(node_idx)
        founds.append(key in next_node.keys)
        node = next_node
    '''
    '''
    return nodes, idxes, founds

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

def empty_node_idx(children):
    for idx, child in enumerate(children):
        if is_empty(child.keys):
            return idx


def _delete(node, key, max_n):
    print('---- go deep ----'); pprint(tuple(node))
    if node.is_leaf:
        idx = index(node.keys, key) # idx in node.keys
        return node._replace(
            keys =(node.keys if idx is None else
                tup_omit(node.keys, idx)))
    else:
        idx = bisect(node.keys, key)
        print('  keys', node.keys, 'key', key, 'idx', idx)
        children = tup_update(
            node.children,
            idx,
            _delete(node.children[idx], key, max_n))
        
        empty_idx = empty_node_idx(children)
        # no steal, no merge
        if empty_idx is None:
            return node._replace(children = children)
        
        empty_node = children[empty_idx]
        new_keys = node.keys
        
        sib_idxes = sibling_idxes(children, empty_idx)
        victim, victim_idx, up_key = theft_victim(
            children, sib_idxes, empty_idx)
        
        if victim: # steal
            key_idx =(
                empty_idx if empty_idx + 1 == victim_idx else
                victim_idx)
            down_key = node.keys[key_idx]
            # build new keys
            new_keys = tup_update(
                new_keys, key_idx, up_key)
            # build new children
            child_lst = list(children)
            child_lst[empty_idx] = empty_node._replace(
                keys = (down_key,))
            child_lst[victim_idx] = victim
            children = tuple(child_lst)
        else: # merge
            #print(' -- node -- '); pprint(tuple(node))
            #print(' -- children -- '); pprint(children)
            sib_idx = sib_idxes[0]
            sibling = children[sib_idx]
            
            key_idx = empty_idx - (1 if empty_idx > 0 else 0)
            #print(' -- sibling -- ')
            #pprint(sibling)
            #print(' -- empty_node -- ')
            #pprint(empty_node)
            #print('sib_idx', sib_idx)
            new_empty_node = empty_node._replace(
                keys = (
                    sibling.keys[0], new_keys[key_idx]
                ) if empty_idx > 0 else (
                    new_keys[key_idx], sibling.keys[0]
                ),
                children = tup_insert(
                    empty_node.children,
                    sib_idx,
                    *sibling.children
                )
            )
            #print(' -- new_empty_node -- ')
            #pprint(new_empty_node)
            
            # build new keys
            new_keys = tup_omit(new_keys, key_idx)
            # build new children
            children = tup_update(
                children, empty_idx, new_empty_node)
            children = tup_omit(
                children, sib_idx)
            
            #print(' -- children after -- ')
            #pprint(children)
            #print(' -- -- -- --')
            
        ret = node._replace(
            keys = new_keys,
            children = children
        )

        print('---- upward ----'); pprint(tuple(ret))
        return ret

    
# minimum number of keys = 0 in 2-3 and 2-3-4 tree.
# it is implicitly implemented.
def delete(tree, key, max_n):
    # Get path root to leaf
    nodes, path, founds = get_path(tree, key)
    #print('---- nodes ----'); pprint(nodes)
    print('---- path ----'); pprint(path)
    # print('---- founds ----'); pprint(founds)
    #print('-- origin --'); pprint(tuple(tree))
    if not any(founds):
        return tree
    if founds[-1] is False: # not leaf
        # remove leftmost key from right node
        r_node = nodes[-1] 
        mv_key = r_node.keys[0]
        new_r_node = r_node._replace(
            keys = tup_omit(r_node.keys, 0))
        # make new tree
        found_depth = founds.index(True)
        found_node = nodes[found_depth]
        new_found = found_node._replace(
            keys = tup_update(
                tree.keys, path[found_depth] - 1, mv_key))
        #print('-- found:', founds)
        #print('-- path:', path)
        #print('-- found path:', found_path)
        #print('-- found depth:', found_depth)
        #print('-- found node:', found_node)
        
        # update tree
        found_path = path[:found_depth]
        tree = update(tree, found_path, new_found)
        tree = update(tree, path, new_r_node)
        # update key
        key = mv_key
        #print('-- replaced --'); pprint(tuple(tree))

    ret = _delete(tree, key, max_n)
    if is_empty(ret.keys):
        if ret.children:
            assert len(ret.children) == 1
            return ret.children[0]
        else:
            return None
    else:
        return ret
    '''
    if not any(founds): # if key not found
        return tree

    rev_seq = reversed(list(zip(nodes, path, founds)))
    for i, (node, node_idx, found) in enumerate(rev_seq):
        node = node_idx
        if node.is_leaf:
            node_key_idx = index(node.keys, key) # idx in node.keys
            new_node = node._replace(
                keys = tup_omit(node.keys, node_key_idx))
            tree = update(tree, idxes[1:], new_node)
        else:
            #pprint(tuple(node))
            pprint(tuple(tree))
            empty_idx = empty_node_idx(node.children)
            if empty_idx is not None: # merge or steal
                empty_child = node.children[empty_idx]
                sib_idxes = sibling_idxes(
                    node.children, empty_idx)
                victim, victim_idx, victim_key_idx = theft_victim(
                    node.children, sib_idxes, empty_idx)
                # make key
                print('make key')
                pprint(tuple(victim))
                pprint(victim_idx)
                pprint(victim_key_idx)
                if victim is not None:
                    node_key_idx =(
                        node_idx if node_idx + 1 == victim_idx else
                        node_idx - 1 if node_idx - 1 == victim_idx else
                        None) # it crashes!
                    node_key = node.keys[node_key_idx]
                    stolen_key = victim.keys[victim_key_idx]
                    # make updated
                    new_empty = empty_child._replace(
                        keys = (node_key,))
                    new_victim = victim._replace(
                        keys = tup_omit(victim.keys, victim_key_idx))
                    # make new node
                    children = list(node.children)
                    children[empty_idx] = new_empty
                    children[victim_idx] = new_victim
                    new_node = node._replace(
                        keys = tup_update(
                            node.keys, node_key_idx, stolen_key),
                        children = tuple(children))
                    # Update tree
                    tree = update(tree, idxes[:-i], new_node)

            
    return tree
    '''
    '''
    rev_nodes = [*reversed(nodes)]
    rev_idxes = [*reversed(idxes), None]
    for node, node_idx in zip(rev_nodes, rev_idxes):
        if node.is_leaf:
            node_key_idx = index(node.keys, key) # idx in node.keys
            new_node = node._replace(
                keys =(node.keys if node_key_idx is None else
                    tup_omit(node.keys, node_key_idx))
            )
            # ---- Make valid b-tree, replace, steal, merge ----
            if is_empty(new_node.keys): # merge or steal
                parent = nodes[-2] # TODO: not only -2
                parent_idx = None # TODO: need?

                sib_idxes = sibling_idxes(parent.children, node_idx)
                victim, victim_idx, victim_key_idx = theft_victim(
                    parent.children, sib_idxes, node_idx)

                assert sib_idxes, 'It cannot be empty'
                if victim is None: # merge
                    #print('-- b --')
                    #pprint(tuple(tree))
                    tree = merge(
                        tree, idxes, node, node_idx, parent, sib_idxes)
                    #print('-- a --')
                    #pprint(tuple(tree))
                    #print('-------')
                else: # steal
                    parent_key_idx =(
                        node_idx if node_idx + 1 == victim_idx else
                        node_idx - 1 if node_idx - 1 == victim_idx else
                        None) # it crashes!
                    tree = steal(
                        tree, idxes, new_node,
                        parent, parent_idx, parent_key_idx,
                        victim, victim_idx, victim_key_idx)
            else:
                tree = update(tree, idxes, new_node)
    return tree
    '''

        #pprint(tuple(update(tree, idxes, new_node)))

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
#max_n = 2; tree = btree(max_n, 4,5,7,8,10)
#max_n = 2; tree = btree(max_n, *range(5,80,5))
#max_n = 2; tree = btree(2, 2,5,7,8)
#max_n = 2; tree = btree(max_n, 4,5,7,8,10)
#max_n = 2; tree = btree(max_n, 0,1)
#max_n = 2; tree = btree(max_n, 0,2,1)
#keys = [1,0,2]; rm_keys = [1,2,0]; max_n = 2; tree = btree(max_n, *keys)
#keys,rm_keys = [0, 1, 2, -1, -2, 3, 4], [-2, 2, 1, 0, 3, -1, 4]
#keys, rm_keys = [0, 1, -1, 2, -2, 3, -3], [-2, -1, 1, 0, 3, 2, -3]
keys, rm_keys = [0, 1, -1, 2, -2, -3, -4, -5], [-2, 1, -3, -1, 0, 2, -5, -4]
#keys, rm_keys = [0, 1, -1, 2, -2, 3, -3, 4, 5], [4, 3, 1, 2, -2, -1, 0, 5, -3]
max_n = 2; tree = btree(max_n, *keys)

print('-------- before --------')
pprint(tuple(tree))
for beg,key in enumerate(rm_keys):
    #pprint(tuple(tree) if tree is not None else tree)
    print('==== rm key:', key, '====', rm_keys[beg + 1:])
    tree = delete(tree, key, max_n)
    pprint(tuple(tree) if tree is not None else tree)
    pprint('==========')
    assert_valid(tree, max_n, tuple(rm_keys[beg + 1:]))
'''
'''

    
'''
print('-------- after --------')
#print(get_path(tree, 40)[1])
tree = delete(tree, 1, max_n)
pprint(tuple(tree))
assert_valid(tree, max_n, rm_keys[1:])
tree = delete(tree, 2, max_n)
pprint(tuple(tree))
assert_valid(tree, max_n, rm_keys[2:])
tree = delete(tree, 0, max_n)
pprint(tree)
assert_valid(tree, max_n, rm_keys[3:])



print('=======================')
keys = (100, 50, 150, 200, 120)
for max_n in [2,3]:
    keys = tuple(keys)
    tree = Node(True, keys[:1])
    print('-------------------======')
    for end,key in enumerate(keys[1:], start=2):
        tree = insert(tree, key, max_n)
        #print(max_n); print(key); print(tree); print(len(keys) > max_n)
        assert_valid(tree, max_n, keys[:end])
'''        
'''
    rm_keys = list(keys)[:]
    random.shuffle(rm_keys)
    for beg,key in enumerate(rm_keys):
        print('---- rm key:', key, '----', rm_keys[beg + 1:])
        tree = delete(tree, key, max_n)
        pprint(tuple(tree))
        assert_valid(tree, max_n, tuple(rm_keys[beg + 1:]))
    '''

'''
max_n = 2; tree = btree(max_n, *range(1,8))
print('-------- before --------')
pprint(tuple(tree))
print('-------- after --------')
pprint(tuple(delete(tree, 3, max_n)))



max_n = 2; tree = btree(2, 2,5,7,8)
max_n = 2; tree = btree(max_n, 3,5,7,9,11,13,15,17)
print('-------- before --------')
pprint(tuple(tree))
print('-------- after --------')
#pprint(tuple(delete(tree, 2,2)))
pprint(tuple(delete(tree, 11,2)))

tree = btree(2, 2,5,7,8)
tree = btree(2, 5,7,8,6)
max_n = 3; tree = btree(max_n, 3,5,7,9,11,13,15,17)
max_n = 2; tree = btree(max_n, 4,5,7,8,10)
max_n = 2; tree = btree(max_n, 1,2,3)
pprint(tuple(delete(tree, 1, max_n)))
#pprint(tuple(delete(tree, 3, max_n)))




keys = [5,10,15,20,25,30]
max_n = 2; tree = btree(max_n, *keys)

shuffled = keys[:]
random.shuffle(shuffled)

pprint(tuple(tree))
for key in shuffled:
    print(f'------ rm:{key} ------')
    pprint(tuple(tree))
    tree = delete(tree, key, max_n)


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
