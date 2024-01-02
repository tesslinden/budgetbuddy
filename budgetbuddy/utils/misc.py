import inspect
from datetime import datetime

import re

def get_arguments(function, locals):#not sure what types to put for these
    """Returns a dictionary of the argument names and their current values for a given function.
    Always pass locals() as the second argument to get_arguments()."""
    # argument_names = function.__code__.co_varnames[:function.__code__.co_argcount] # this might also work
    argument_names = list(inspect.signature(function).parameters.keys())
    argument_values = [locals[argument_name] for argument_name in argument_names if argument_name in locals.keys()]
    arguments = dict(zip(argument_names, argument_values))
    return arguments

def get_yrmo(dt: datetime):
    """Takes a datetime object and returns a string in the format "YYYY-MM".
    """
    return dt.strftime('%Y-%m')

def as_concise_yrmo(yrmo: str):
    """Takes a yrmo string (a datetime string in the format "YYYY-MM") and returns a string in the format "M/YY".
    """
    assert re.match(r'\d{4}-\d{2}', yrmo), "yrmo must be in format YYYY-MM"
    concise_yrmo = f"{int(yrmo[5:])}/{yrmo[2:4]}"
    return concise_yrmo