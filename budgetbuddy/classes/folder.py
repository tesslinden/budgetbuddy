from dataclasses import dataclass
from typing import Dict, Optional

from .. import config


@dataclass
class Folder:
    name: str
    prefix: str
    suffix: str
    final_column_index: Optional[int] = None
    columns_to_rename: Optional[Dict[str,str]] = None
    account: Optional[str] = None
    reformat_amounts_alliant: bool = False #TODO: replace this with a more robust, not-hardcoded solution. probably there is a python package that does it automatically.
    flip_sign_of_amounts: bool = False
    prune_columns: bool = False
    fill_false: bool = False
    filename_format: Optional[str] = None

    @property
    def path(self):
        return f"{config.PATH_TO_TRANSACTIONS}/{self.name}"