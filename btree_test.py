import pytest
from hypothesis import given, example
from hypothesis import strategies as st

import funcy as F

from btree import *


def test_insert_no_split():
    max_n = 2
    assert insert(leaf(10), 20, 2) == Node(True, (10, 20), ())
    assert insert(leaf(20), 10, 2) == Node(True, (10, 20), ())
    
    max_n = 3
    tree = insert(leaf(10), 20, 3)
    assert tree == Node(True, (10, 20), ())
    tree = insert(tree, 0, 3)
    assert tree == Node(True, (0, 10, 20), ())

def test_insert_split():
    max_n = 2
    ins2 = lambda *args: insert(*args, 2)
    root = ins2(leaf(5), 6)
    root = ins2(root, 7)
    assert root == Node(False, (6,), (leaf(5), leaf(7)))
    
    max_n = 3
    ins3 = lambda *args: insert(*args, 3)
    #root = insert(leaf(5), 6, 2)
    #root = insert(root, 7, 2)

#@given(st.lists(st.integers(), min_size = 1))
#def test_insert_prop_test(xs):
def test_insert_prop_test():
    # number of keys inserted/flattend btree are same
    # all nodes are valid
    
    # keys are sorted from dfs
    # all leaves have same height
    tree = Node(False, (6,), (leaf(5), leaf(7)))
    print(Node(False, (6,), (leaf(5), leaf(7))))
    print(F.lflatten(tree))
    assert False

@pytest.mark.skip(reason="no way of currently testing this")
def test_split_manually():
    max_n = 3
    full_child = Node(True, (11,12,13))
    unfull_parent = Node(
        False, (10,), (full_child, Node(True, (20,))))

    actual = split_child(unfull_parent, 0,)
    expect = Node(False, (10,),
                   (Node(True, (11,)), Node(True, (13,))))
    assert not is_invalid(unfull_parent)
    assert not is_invalid(full_child)
    assert not is_invalid(actual)
    assert not is_invalid(expect)
    assert actual == expect, rep(actual)+' != '+rep(expect)
        
