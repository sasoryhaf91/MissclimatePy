# tests/test_imports.py
def test_imports():
    import missclimatepy
    from missclimatepy import (
        neighbors, masking
    )

    assert hasattr(missclimatepy, "__version__")
    assert callable(neighbors.build_neighbor_map)
    assert callable(masking.percent_missing_between)
