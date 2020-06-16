from collections import namedtuple


Node = namedtuple(
    'Node',
    'is_leaf min_n max_n keys children', defaults=[(), ()])

def Node3(is_leaf, *args): # 3 means max num of keys in node
    return Node(is_leaf, 1, 3, *args) # 2-3-4
def Node2(is_leaf, *args): # 2 means max num of keys in node
    return Node(is_leaf, 1, 2, *args) # 2-3

root = Node3(True)
print(root)
