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
    return Node(is_leaf, 1, 2, keys, children) # 2-3
def Node3(is_leaf, keys, children=()): # 3 = max num of keys
    return Node(is_leaf, 1, 3, keys, children) # 2-3-4

def split_child(unfull_parent, child_index):
    pass

print(rep(Node3(False, (10, 12, 20),
                   (Node3(True, (11,)), Node3(True, (13,))))))
print(type(Node3(False, (10, 12, 20),
                   (Node3(True, (11,)), Node3(True, (13,))))))
