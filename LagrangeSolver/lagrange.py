from sympy import Symbol, diff, solve, Add, UnevaluatedExpr
from utils import drop_lambda

def _sum(l):
    return Add(*l)

sum = _sum

TQDM_MONITOR = True
PRINT_MONITOR = True

if TQDM_MONITOR:
    from tqdm import tqdm
else:
    tqdm = lambda x: x

if not PRINT_MONITOR:
    print = lambda _: None

def lagrange_system(obj, cons, symbols):
    syms = drop_lambda(symbols)
    grad_obj = [diff(obj, i) for i in tqdm(syms)]
    grad_cons = [symbols["lambda"] * diff(cons, i) for i in tqdm(syms)]
    return grad_obj, grad_cons

def solve_grads(grad_obj, grad_cons, cons):
    return solve([o - c for o, c in zip(grad_obj, grad_cons)] + [cons], dict=True)

def manual_solve_grads(grad_obj, grad_cons, cons, symbols):
    solved = {}
    for sym, go, gc in tqdm(list(zip(drop_lambda(symbols), grad_obj, grad_cons))):
        for s, e in solved.items():
            go = go.subs(s, e)
            gc = gc.subs(s, e)
        sym_solved = solve(go - gc, sym)[0]
        for s in solved:
            solved[s] = solved[s].subs(sym, sym_solved)
        solved[sym] = sym_solved
    for s, e in solved.items():
        cons = cons.subs(s, e)
    l = symbols["lambda"]
    solved[l] = solve(cons, l)[0]
    for s in solved:
        solved[s] = solved[s].subs(l, solved[l])
    return solved

def apply_solution(sol, teams):
    return {t: sol[tsym] for t, tsym in teams.items()}

def run_strat(symbols, obj, cons):
    grad_obj, grad_cons = lagrange_system(obj, cons, symbols)
    sol = manual_solve_grads(grad_obj, grad_cons, cons, symbols)
    return apply_solution(sol, symbols)