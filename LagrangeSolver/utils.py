def jl(*args):
    """join lists recursively"""
    if len(args) == 0:
        return []
    return args[0] + jl(*args[1:])

def drop_lambda(syms):
    return [syms[i] for i in syms.keys() if i != "lambda"]