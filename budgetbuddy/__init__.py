try:
    from .. import config
except ImportError:
    raise ImportError(
        "config.py not found. Please ensure config.py exists in the budgetbuddy directory. \
        You can create this file by copying from config.py.example and updating it with your \
        settings."
    )
