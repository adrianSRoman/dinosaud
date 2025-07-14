import torch

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

def filter_by_keys(fn, d):
    return {k: v for k, v in d.items() if fn(k)}

def map_keys(fn, d):
    return {fn(k): v for k, v in d.items()}
