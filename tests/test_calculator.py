import pytest

from tools.calculator import calculate


def test_basic_arithmetic():
    assert calculate("2 + 3 * 4") == "14"


def test_growth_rate_expression():
    assert calculate("(150 - 100) / 100 * 100") == "50.0"


def test_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        calculate("1 / 0")


def test_invalid_syntax_raises():
    with pytest.raises(Exception):
        calculate("2 +")


def test_unknown_name_raises():
    with pytest.raises(Exception):
        calculate("foo + 1")


def test_rejects_dunder_attribute_access():
    with pytest.raises(Exception):
        calculate("().__class__")


def test_rejects_import_call():
    with pytest.raises(Exception):
        calculate("__import__('os').system('echo hi')")


def test_returns_string():
    assert isinstance(calculate("1"), str)


