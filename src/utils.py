import glob
import os
import random
import string
from pathlib import Path
from typing import Union


def generate_unique_file_name(folder: Union[Path, str], extension: str) -> Path:
    def _generate_uid(length: int = 10) -> str:
        assert 0 < length < 100
        return "".join(
            random.choice(string.ascii_lowercase + string.digits) for _ in range(length)
        )

    if isinstance(folder, str):
        folder = Path(folder)

    temp_uid = _generate_uid()
    while (folder / Path(temp_uid + extension)).exists():
        temp_uid = _generate_uid()

    return folder / Path(temp_uid + extension)


def cleanup(file_path_without_extension: Union[str, Path]):
    file_path_without_extension = Path(file_path_without_extension)
    matching_files = glob.glob(str(file_path_without_extension.as_posix()) + '*')
    for file in matching_files:
        os.remove(file)
