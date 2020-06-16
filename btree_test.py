from hypothesis import given, example
from hypothesis import strategies as st

from btree import *


def test_split_manually():
    full_child = Node3(True, (11,12,13))
    unfull_parent = Node3(
        False, (10,), (full_child, Node3(True, (20,))))

    actual = split_child(unfull_parent, 0)
    expect = Node3(False, (10,),
                   (Node3(True, (11,)), Node3(True, (13,))))
    assert not is_invalid(unfull_parent)
    assert not is_invalid(full_child)
    assert not is_invalid(actual)
    assert not is_invalid(expect)
    assert actual == expect, rep(actual)+' != '+rep(expect)
        
