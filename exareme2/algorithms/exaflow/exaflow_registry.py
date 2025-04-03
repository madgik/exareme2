from exareme2.utils import Singleton


class ExaflowRegistry(metaclass=Singleton):
    def __init__(self):
        self._registry = {}

    def register(self, func):
        """
        Registers a function into the registry with a unique key.

        This method generates a unique key by combining the last segment of the function's module name
        with the function's name (formatted as "module_functionName"). It then stores the function in the
        registry dictionary under this key, allowing for easy retrieval later.

        Parameters:
            func (function): The user-defined function to be registered.

        Returns:
            function: The original function passed in, which makes this method suitable for use as a decorator.

        Example:
            @exaflow_udf
            def my_udf(x):
                return x * 2

            # If defined in a module named "analytics", the function is registered with the key "analytics_my_udf".
        """
        key = f"{func.__module__.split('.')[-1]}_{func.__name__}"
        self._registry[key] = func
        return func

    def get_udf(self, name):
        return self._registry.get(name)

    def get_available_udfs(self):
        return list(self._registry.keys())


# Singleton instance
exaflow_registry = ExaflowRegistry()


# Decorator to simplify registration
def exaflow_udf(func):
    return exaflow_registry.register(func)
