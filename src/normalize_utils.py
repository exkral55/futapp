import pandas as pd

def standardize_name(name):
    if pd.isna(name):
        return ""
    n = name.lower()
    n = n.replace("ı","i").replace("ğ","g").replace("ş","s")
    n = n.replace("ö","o").replace("ü","u").replace("ç","c")
    n = "".join(ch for ch in n if ch.isalnum() or ch==" ")
    return " ".join(n.split())