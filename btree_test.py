from hypothesis import given, example
from hypothesis import strategies as st

from btree import *


def test_split_manually():
    full_child = Node3(True, (11,12,13))
    unfull_parent = Node3(False, (10, 2), (full_child,))

    actual = split_child(unfull_parent, 0)
    expect = Node3(False, (10, 12, 20),
                   (Node3(True, (11,)), Node3(True, (13,))))
    assert actual == expect, f'{rep(actual)} != {rep(expect)}'
        
