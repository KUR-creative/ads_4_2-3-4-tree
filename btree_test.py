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

def test_steal_what():
    pass
#victim, victim_idxes, key_idx

@given(st.lists(
    st.integers(min_value=0, max_value=3),
    min_size=1, max_size=4
).flatmap(
    lambda xs: st.tuples(
        st.just(xs),
        st.integers(min_value=-1, max_value=len(xs) - 1))
))
def test_sibling_idxes(xs_idx):
    xs, idx = xs_idx
    sib_idxes = sibling_idxes(xs, idx)
    ans_idx = idx + len(xs) if idx < 0 else idx
    for sidx in sib_idxes:
        xs[sidx] # assert no crash
        assert ans_idx == sidx - 1 or ans_idx == sidx + 1

#@pytest.mark.skip(reason="not now")
def test_delete_explicit_sequence():
    max_n = 2
    # no found
    tree = btree(2, 2,5,7,8)
    assert tree == delete(tree, -100, 2)
    
    # leaf(replace x), steal x, merge x
    assert(delete(tree, 7, 2) 
        == Node(False, (5,), (
            Node(is_leaf=True, keys=(2,), children=()),
            Node(is_leaf=True, keys=(8,), children=()))))
    # leaf(replace x), steal (L <- R), merge x
    assert(delete(tree, 2, 2) 
        == Node(False, (7,), (
            Node(is_leaf=True, keys=(5,), children=()),
            Node(is_leaf=True, keys=(8,), children=()))))
    # leaf(replace x), steal (L -> R), merge x
    assert(delete(btree(2, 5,7,8,6), 8, 2)
        == Node(False, (6,), (
            Node(is_leaf=True, keys=(5,), children=()),
            Node(is_leaf=True, keys=(7,), children=()))))
    # leaf, steal (L -> R, parent many keys), merge x
    max_n = 3; tree = btree(max_n, 3,5,7,9,11,13,15,17)
    assert(delete(tree, 11, max_n)
        == Node(False, (5, 9, 15), (
            Node(is_leaf=True, keys=(3,), children=()),
            Node(is_leaf=True, keys=(7,), children=()),
            Node(is_leaf=True, keys=(13,), children=()),
            Node(is_leaf=True, keys=(17,), children=()))))
    
    # leaf(replace x), steal x, merge o
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
