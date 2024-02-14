import inspect
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import re

def as_concise_yrmo(yrmo: str) -> str:
    """Takes a yrmo string (a datetime string in the format "YYYY-MM") and returns a string in the format "M/YY".
    """
    assert re.match(r'\d{4}-\d{2}', yrmo), "yrmo must be in format YYYY-MM"
    concise_yrmo = f"{int(yrmo[5:])}/{yrmo[2:4]}"
    return concise_yrmo

def get_arguments(function: Callable, locals: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a dictionary of the argument names and their current values for a given function.
    Always pass locals() as the second argument to get_arguments().
    """
    # argument_names = function.__code__.co_varnames[:function.__code__.co_argcount] # this might also work
    argument_names = list(inspect.signature(function).parameters.keys())
    argument_values = [locals[argument_name] for argument_name in argument_names if argument_name in locals.keys()]
    arguments = dict(zip(argument_names, argument_values))
    return arguments

def get_yrmo(dt: datetime) -> str:
    """Takes a datetime object and returns a string in the format "YYYY-MM".
    """
    return dt.strftime('%Y-%m')

def swap_positions_in_list(l: list, item1: Any, item2: Any) -> list:
    """Swaps the positions of two elements in a list and returns the modified list.
    """
    assert item1 in l and item2 in l, "Both items must be in the list"
    assert isinstance(l, list), "l must be a list"
    index1 = l.index(item1)
    index2 = l.index(item2)
    l[index1], l[index2] = l[index2], l[index1]
    return l