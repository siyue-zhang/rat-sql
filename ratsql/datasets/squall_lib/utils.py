import unicodedata
import re


def normalize(x):
    """ Normalize string. """
    if x is None:
        return None
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub("[‘’´`]", "'", x)
    x = re.sub("[“”]", "\"", x)
    x = re.sub("[‐‑‒–—−]", "-", x)
    # Replace \n by space
    x = x.replace('\n', ' ')
    # Collapse whitespaces and convert to lower case
    x = re.sub('\s+', ' ', x, flags=re.U)
    # Replace signal for brackets
    x = x.replace("-lrb-","(").replace("-rrb-",")").lower().strip()
    return x

if __name__ == "__main__":
    x = "-lrb-"
    print(x)
    print(normalize(x))