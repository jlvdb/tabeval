import pytest

import numpy as np
import numpy.testing as npt
import tabeval


@pytest.fixture
def dataset():
    return dict(
        x=np.arange(10),
        y=np.arange(10, 20),
        is_short=np.arange(9),
        is_string=np.array(list(str(i) for i in range(10, 20))),
        is_list=list(range(10)))


@pytest.mark.parametrize(
    "value", [2, 0.4, "somestring"])
def test_identity(value):
    term = tabeval.MathTerm.from_string(str(value))
    if type(value) is str:
        with pytest.raises(ValueError):
            term()
    else:
        assert term() == value


@pytest.mark.parametrize(
    "value", [2, 0.4, "somestring"])
def test_identity_with_data(value, dataset):
    term = tabeval.MathTerm.from_string(str(value))
    if type(value) is str:
        with pytest.raises(KeyError):
            term(dataset)
    else:
        assert term(dataset) == value


@pytest.mark.parametrize(
    "math_string", [
        "(2*(4**3)*x + 2",
        "y*(4**y)*6789 + 2)**0.4",
        "(y +* x)",
        "1 * (x*(4 + AND y)*x + x)**0.4",
        "| 1 * (2*(y**3)*6789 + y)**0.4",
        "== y * (x*(4**3)*6789 + 2)+-0.4",
        "2 * (((y + 2",
])
def test_syntax_errors(math_string):
    with pytest.raises(SyntaxError):
        tabeval.MathTerm.from_string(math_string)


@pytest.mark.parametrize(
    "math_string", [
        "(2*(4**3)*6789 + 2)**0.4",
        "(2 + 2)",
        "1 * (2*(4 + 3)*6789 + 2)**0.4",
        "1 * (2*(4**3)*6789 + 2)**0.4",
        "-1 * (2*(4**3)*6789 + 2)+-0.4",
        "1 * (2*(4**3)*6789 + 2)+0.4",
        "2 * (((1 + 2)))",
        "2 * ((4) + 5)",
        "2 * (1 + 2)",
        "2 * (4 % 5)",
        "2 * 1 + 2",
        "2*((4**3))*6789 + 2",
        "2*(4**3)*6789 + 2 +2",
        "2*(4**3)*6789 + 2",
        "2*(4**3)*6789.3 + 2 +2",
        "5 / ((2 * 2) + 1)",
        "5 / ((2 * 2) + 2)",
        "5 / ((2 * 2))",
        "5 / (2 * (4 + 5))",
        "5 / (2 * 1)",
        "5 | (2 ^ 2)",
        "5>=(2-3)",
        "-4"
])
def test_scalar(math_string, dataset):
    term = tabeval.MathTerm.from_string(math_string)
    result = eval(math_string)
    assert term() == result
    assert term(dataset) == result


@pytest.mark.parametrize(
    "math_string", [
        "(2*(4**3)*x + 2)",
        "(y*(4**y)*6789 + 2)**0.4",
        "(y + x)",
        "-1 * (x*(4 + y)*x + x)**0.4",
        "1 * (2*(y**3)*6789 + y)**0.4",
        "y * (x*(4**3)*6789 + 2)+-0.4",
        "1 * (2*(4**3)*y + x)+x",
        "2 * (((y + 2)))",
        "2 * ((x) + y)",
])
def test_vector(math_string, dataset):
    term = tabeval.MathTerm.from_string(math_string)
    python_string = math_string\
        .replace("x", "dataset['x']")\
        .replace("y", "dataset['y']")
    result = eval(python_string)
    npt.assert_array_equal(term(dataset), result)


def test_vector_bad_values(dataset):
    term = tabeval.MathTerm.from_string("x - is_string")
    with pytest.raises(TypeError):
        term(dataset)


def test_vector_wrong_length(dataset):
    term = tabeval.MathTerm.from_string("x - is_short")
    with pytest.raises(ValueError):
        term(dataset)


def test_list_variables():
    term = tabeval.MathTerm.from_string("a *(x - 2) + b - alt / o**2")
    assert term.list_variables() == ["a", "alt", "b", "o", "x"]


@pytest.mark.parametrize(
    "math_string,expression_string", [
        ("(2*(4**3)*6789 + 2)**0.4",        "(((2 * (4 ** 3)) * 6789) + 2) ** 0.4"),
        ("(2 + 2)",                         "2 + 2"),
        ("1 * (2*(4 + 3)*6789 + 2)**0.4",   "1 * ((((2 * (4 + 3)) * 6789) + 2) ** 0.4)"),
        ("1 * (2*(4**3)*6789 + 2)**0.4",    "1 * ((((2 * (4 ** 3)) * 6789) + 2) ** 0.4)"),
        ("-1 * (2*(4**3)*6789 + 2)+-0.4",   "((- 1) * (((2 * (4 ** 3)) * 6789) + 2)) + (- 0.4)"),
        ("1 * (2*(4**3)*6789 + 2)+0.4",     "(1 * (((2 * (4 ** 3)) * 6789) + 2)) + 0.4"),
        ("2 * (((1 + 2)))",                 "2 * (1 + 2)"),
        ("2 * ((4) + 5)",                   "2 * (4 + 5)"),
        ("2 * (1 + 2)",                     "2 * (1 + 2)"),
        ("2 * (4 % 5)",                     "2 * (4 % 5)"),
        ("2 * 1 + 2",                       "(2 * 1) + 2"),
        ("2*((4**3))*6789 + 2",             "((2 * (4 ** 3)) * 6789) + 2"),
        ("2*(4**3)*6789 + 2 +2",            "(((2 * (4 ** 3)) * 6789) + 2) + 2"),
        ("2*(4**3)*6789 + 2",               "((2 * (4 ** 3)) * 6789) + 2"),
        ("2*(4**3)*6789.3 + 2 +2",          "(((2 * (4 ** 3)) * 6789.3) + 2) + 2"),
        ("5 / ((2 * 2) + 1)",               "5 / ((2 * 2) + 1)"),
        ("5 / ((2 * 2) + 2)",               "5 / ((2 * 2) + 2)"),
        ("5 / ((2 * 2))",                   "5 / (2 * 2)"),
        ("5 / (2 * (4 + 5))",               "5 / (2 * (4 + 5))"),
        ("5 / (2 * 1)",                     "5 / (2 * 1)"),
        ("5 | (2 ^ 2)",                     "5 | (2 ^ 2)"),
        ("5>=(2-3)",                        "5 >= (2 - 3)"),
        ("-4",                              "- 4"),
        ("1",                               "ID 1"),
])
def test_expression(math_string, expression_string):
    term = tabeval.MathTerm.from_string(math_string)
    assert term.expression == expression_string
