# tabeval

Python eval() on key-value-based numerical data structures

## Example

```python
from tabeval import MathTerm, evaluate

data = dict(x=[1, 2, 3])  # some test data

term = MathTerm.from_string("2*x >= 4")
print(term.expression)    # math representation
print(term.expression)    # code representation
print(term(data))         # evaluate on dataset: False, True, True

print(evaluate("2*x >= 4", data))  # short form
```

## Installation

Can be installed with `python setup.py` or  
`pip install "tabeval @ git+https://github.com/jlvdb/tabeval.git"`
