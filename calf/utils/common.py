import unicodedata
from typing import Dict

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
NUL = "<nul>"

MIN = -1e32
INF = float("inf")


def ispunct(token: str) -> bool:
    return all(unicodedata.category(char).startswith('P') for char in token)


def isfullwidth(token: str) -> bool:
    return all(unicodedata.east_asian_width(char) in ['W', 'F', 'A'] for char in token)


def islatin(token: str) -> bool:
    return all("LATIN" in unicodedata.name(char) for char in token)


def isdigit(token: str) -> bool:
    return all("DIGIT" in unicodedata.name(char) for char in token)


def tohalfwidth(token: str) -> str:
    return unicodedata.normalize("NFKC", token)


def get_signature(dataobject: Dict):
    import hashlib
    return hashlib.md5(str(list(dataobject.items())).encode()).hexdigest()
