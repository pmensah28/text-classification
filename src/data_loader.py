import io
from typing import List, Tuple

def read_data(filename: str) -> List[Tuple[str, List[str]]]:
    with io.open(filename, 'r', encoding='utf-8') as file:
        data = []
        for line in file:
            tokens = line.split()
            data.append((tokens[0], tokens[1:]))
    return data
