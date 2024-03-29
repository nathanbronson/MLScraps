from lagrange import run_strat, _sum, drop_lambda
from sympy import Symbol
from random import randint

if __name__ == "__main__":
    symbols = {"x": Symbol("x"), "y": Symbol("y"), "z": Symbol("z"), "lambda": Symbol("lambda")}
    obj = (symbols["x"] + 3)**2 + (symbols["y"] + 2)**2 + (symbols["z"] - 5)**2
    cons = 2 * symbols["x"] + 3 * symbols["y"] + .25 * symbols["z"] - 6
    print(run_strat(symbols, obj, cons))
    symbols = {str(i): Symbol(str(i)) for i in range(100)} | {"lambda": Symbol("lambda")}
    obj = _sum([(i - randint(-10, 10)) ** 2 for i in drop_lambda(symbols)])
    cons = _sum([randint(-10, 10) * i for i in drop_lambda(symbols)] + [randint(-10, 10)])
    print(run_strat(symbols, obj, cons))