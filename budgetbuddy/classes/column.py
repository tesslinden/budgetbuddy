from dataclasses import dataclass

@dataclass
class Column: 
    name: str
    base: bool
    display: bool
    dtype: type