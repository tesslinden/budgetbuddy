import inspect

def get_arguments(function, locals):#not sure what types to put for these
    """Returns a dictionary of the argument names and their current values for a given function.
    Always pass locals() as the second argument to get_arguments()."""
    # argument_names = function.__code__.co_varnames[:function.__code__.co_argcount] # this might also work
    argument_names = list(inspect.signature(function).parameters.keys())
    argument_values = [locals[argument_name] for argument_name in argument_names if argument_name in locals.keys()]
    arguments = dict(zip(argument_names, argument_values))
    return arguments