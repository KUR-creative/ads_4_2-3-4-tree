import random
from pprint import pprint

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
    
def test_insert_explicit_sequence():
    max_n = 2
    #keys = (100, 50, 150, 200, 120, 135, 140, 170)
    #keys = (100, 50, 150, 200, 120, 135, 140, 170, 250)
    keys = (100, 50, 150, 200, 120, 135, 140, 170, 250, 145)
    expecteds = [
        Node(True, (100,), ()),
        Node(True, (50, 100), ()),
        Node(False, (100,),
             (Node(is_leaf=True, keys=(50,), children=()),
              Node(is_leaf=True, keys=(150,), children=()))),
        Node(False, (100,),
             (Node(is_leaf=True, keys=(50,), children=()),
              Node(is_leaf=True, keys=(150, 200), children=()))),
        Node(False, (100, 150),
             (Node(is_leaf=True, keys=(50,), children=()),
              Node(is_leaf=True, keys=(120,), children=()),
              Node(is_leaf=True, keys=(200,), children=()))),
        Node(False, (100, 150),
             (Node(is_leaf=True, keys=(50,), children=()),
              Node(is_leaf=True, keys=(120, 135), children=()),
              Node(is_leaf=True, keys=(200,), children=()))),
        Node(False, (135,),
             (Node(False, (100,), (
                 Node(is_leaf=True, keys=(50,), children=()),
                 Node(is_leaf=True, keys=(120,), children=()))),
              Node(False, (150,), (
                  Node(is_leaf=True, keys=(140,), children=()),
                  Node(is_leaf=True, keys=(200,), children=()))))),
        Node(False, (135,),
             (Node(is_leaf=False, keys=(100,), children=(Node(is_leaf=True, keys=(50,), children=()), Node(is_leaf=True, keys=(120,), children=()))),
              Node(is_leaf=False, keys=(150,), children=(Node(is_leaf=True, keys=(140,), children=()), Node(is_leaf=True, keys=(170, 200), children=()))))),
        Node(False, (135,),
             (Node(is_leaf=False, keys=(100,), children=(Node(is_leaf=True, keys=(50,), children=()), Node(is_leaf=True, keys=(120,), children=()))),
              Node(is_leaf=False, keys=(150, 200), children=(Node(is_leaf=True, keys=(140,), children=()), Node(is_leaf=True, keys=(170,), children=()), Node(is_leaf=True, keys=(250,), children=()))))),
        Node(False, (135,),
             (Node(is_leaf=False, keys=(100,), children=(Node(is_leaf=True, keys=(50,), children=()), Node(is_leaf=True, keys=(120,), children=()))),
              Node(is_leaf=False, keys=(150, 200), children=(Node(is_leaf=True, keys=(140, 145), children=()), Node(is_leaf=True, keys=(170,), children=()), Node(is_leaf=True, keys=(250,), children=())))))
    ]
    tree = Node(True, keys[:1])
    for key, expected in zip(keys[1:], expecteds[1:]):
        print('---- inp:', key, '----')
        tree = insert(tree, key, max_n)
        assert tree == expected

#---------------------------------------------------------
@st.composite
def gen_tup(draw):
    tup = tuple(draw(st.lists(st.integers(), min_size=1)))
    idx = random.randint(0, len(tup) - 1)
    new = draw(st.integers())
    return [tup, idx, new]

@given(gen_tup())
def test_tuple_update(tup_idx_new):
    tup,idx,new = tup_idx_new
    new_tup = tuple_update(tup, idx, new)
    print(new_tup)
    assert new_tup[idx] == new

@given(gen_tup())
def test_tuple_omit(tup_idx_new):
    tup, idx, _ = tup_idx_new
    new_tup = tuple_omit(tup, idx)
    assert tuple_insert(new_tup, idx, tup[idx]) == tup
    
#---------------------------------------------------------
@given(st.integers(min_value=2, max_value=3),
       st.lists(st.integers(), min_size = 2, unique=True))
def test_btree(max_n, keys):
    tree = btree(max_n, *keys)
    print(tree)
    assert_valid(tree, max_n, keys)
#@pytest.mark.skip(reason="useless")
@given(st.lists(st.integers(), min_size = 2, unique=True))
def test_insert_prop_test_max2(keys):
    for max_n in [2,3]:
        keys = tuple(keys)
        tree = Node(True, keys[:1])
        print('-------------------======')
        for end,key in enumerate(keys[1:], start=2):
            tree = insert(tree, key, max_n)
            print(max_n); print(key); print(tree); print(len(keys) > max_n)
            assert_valid(tree, max_n, keys[:end])
    
@pytest.mark.skip(reason="useless")
def test_all_keys_all_nodes():
    
    #tree = Node(False, (6,), (leaf(5), leaf(7)))
    print('===rr=====')
    #tree = Node(False, (5,10), (leaf(0,1), leaf(7,8), leaf(15)))
    tree = Node(is_leaf=False,
        keys=(1,),
        children=(Node(is_leaf=True,
                        keys=(0,),
                        children=()),
                Node(is_leaf=True,
                        keys=(1,),
                        children=())))
    ks = all_keys(tree)
    ns = all_nodes(tree)
    print(ks)
    #print([n.keys for n in ns])
    #pprint(ns)
    pprint([n.keys for n in ns])
    print(len(ns))
    print('====--------======')
    assert False
    
    tree = Node(
        False, (5,),
        (Node(False, (1,), (leaf(0), leaf(2))),
         Node(False, (8,10), (leaf(7), leaf(9), leaf(15)))))
    ks = all_keys(tree)
    ns = all_nodes(tree)
    print(ks)
    print(len(ns))
    pprint([n.keys for n in ns])
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
        
