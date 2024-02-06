class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __getattr__(self, item):
        return self.__dict__.get(item, None)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __iter__(self):
        return iter(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __getitem__(self, key):
        return self.__dict__.get(key, None)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def copy(self):
        # Create a new instance of Config with the same entries
        return Config(**self.__dict__)


def get_config():
    config = Config(
        epochs=100,
        step_num=100,
        dist_type="L1",
        time_scaling_factor=3.0,
        std_dev_scaling_factor=0.5,
        ddpm_start=0.0001,
        ddpm_end=0.01,
        vectorbridge_magnitude_scale=3.0,
        vectorbridge_rand_scale=3.0,
        vectorbridgeddpm_use_simple_posterior_std=False,
    )
    return config


def get_params_old(func, config, kwargs=None):
    # Determine the function to inspect (regular function, __init__ of class, or __call__ of callable object)
    if not callable(func):
        raise TypeError(f"{func} is not callable")

    if hasattr(func, "__init__") and isinstance(func, type) and func.__init__ != object.__init__:
        # It's a class, inspect the __init__ method
        func_to_inspect = func.__init__
    elif callable(getattr(func, "__call__", None)):
        # It's a callable object, inspect the __call__ method
        func_to_inspect = func.__call__
    else:
        # It's a regular function or something else callable
        func_to_inspect = func

    # Get the code object from the function to inspect
    if hasattr(func_to_inspect, "__code__"):
        code = func_to_inspect.__code__
    else:
        raise TypeError(f"Cannot inspect the callable object {func}. It doesn't have a __code__ attribute.")

    # Exclude 'self' for methods of classes or callable objects
    start_index = 1 if "self" in code.co_varnames[: code.co_argcount] else 0
    valid_keys = code.co_varnames[start_index : code.co_argcount]
    filtered_config = {k: config[k] for k in valid_keys if k in config}

    if kwargs is not None:
        for key, value in kwargs.items():
            filtered_config[key] = value

    return filtered_config


import inspect

# TODO fix this function
def get_params(func, config, kwargs=None):
    if not callable(func):
        raise TypeError(f"{func} is not callable")

    # Use inspect to get the signature
    try:
        sig = inspect.signature(func)
    except ValueError:
        # Handle classes specially since inspect.signature() does not work directly on classes
        if inspect.isclass(func):
            # Try to get signature of __init__ method
            sig = inspect.signature(func.__init__)
        else:
            raise TypeError(f"Cannot inspect the callable object {func}. It doesn't have a __code__ attribute.")

    params = sig.parameters
    valid_keys = [key for key in params if params[key].kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]]

    filtered_config = {k: config[k] for k in valid_keys if k in config}

    if kwargs is not None:
        for key, value in kwargs.items():
            filtered_config[key] = value

    return filtered_config
