from functools import lru_cache
from pathlib import Path
PROMPT_DIR = Path(__file__).resolve().parent

@lru_cache(maxsize= None)
def _read(name): 
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()

def load_prompt(name , **values): 
    text = _read(name)
    for key , value in values.items(): 
        text = text.replace("{" + key + "}" , value)

    return text
