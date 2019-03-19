from dsflow import preprocessing


def test__add_weight():
    tokens = ['a', 'b', 'c']
    weights = [1, 2, 3]
    assert 1 == preprocessing._add_weight('a', tokens, weights)
    assert 2 == preprocessing._add_weight('b', tokens, weights)
    assert 1 == preprocessing._add_weight('z', tokens, weights)
