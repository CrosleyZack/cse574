from functools import wraps
from time import time

def timed(func):
    @wraps(func)
    def wrapper(*argv, **kwargs):
        start = time()
        return_val = func(*argv, **kwargs)
        diff = time() - start
        string = str(func.__name__) + " executed in " + str(diff)
        print(string)
        with open('timing_info.txt', 'a+') as f:
            f.write(f'{string}\n')
        return return_val
    return wrapper