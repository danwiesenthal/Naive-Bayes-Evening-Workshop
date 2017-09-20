import functools

#  Advanced material here, feel free to ignore memoize if you like.  Long story short, it remembers the inputs and outputs to functions, and if the same input is seen multiple times, then rather than running the function multiple times, memoization just returns the result of the function when it was first called with that input.  Memory expensive because it keeps track of past results, but computationally nice because we don't recalculate the same thing over and over.


def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer
