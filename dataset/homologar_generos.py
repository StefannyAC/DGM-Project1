# homologar_generos.py (compat Python 3.8+)

import re
import json
import argparse
import unicodedata
from difflib import get_close_matches
from collections import Counter
from typing import Optional, List, Dict, Tuple

import pandas as pd

COARSE: Dict[str, int] = {"classical": 0, "jazz": 1, "rock": 2, "pop": 3}
PRIORITY: List[str] = ["classical", "jazz", "rock", "pop"]

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[-_/]+", " ", s)
    s = re.sub(r"[^\w\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str) -> List[str]:
    return norm(s).split()

ALIASES: Dict[str, str] = {
    # classical
    "baroque":"classical","romantic":"classical","classical":"classical",
    "orchestral":"classical","symphony":"classical","concerto":"classical",
    "opera":"classical","choral":"classical","sonata":"classical",
    "prelude":"classical","etude":"classical","score":"classical","film score":"classical",
    "neo classical":"classical",
    # jazz
    "jazz":"jazz","bebop":"jazz","hard bop":"jazz","swing":"jazz","ragtime":"jazz",
    "bossa nova":"jazz","bossa":"jazz","fusion":"jazz","big band":"jazz","smooth jazz":"jazz",
    # rock
    "rock":"rock","alt rock":"rock","alternative rock":"rock","indie rock":"rock",
    "progressive rock":"rock","prog rock":"rock","hard rock":"rock","punk":"rock",
    "grunge":"rock","metal":"rock","heavy metal":"rock","death metal":"rock",
    "black metal":"rock","thrash metal":"rock","metalcore":"rock","post rock":"rock",
    "garage rock":"rock","shoegaze":"rock","math rock":"rock",
    # pop
    "pop":"pop","dance pop":"pop","synth pop":"pop","electropop":"pop",
    "r b":"pop","rnb":"pop","soul":"pop","funk":"pop",
    "hip hop":"pop","rap":"pop","trap":"pop",
    "edm":"pop","electronic":"pop","house":"pop","techno":"pop","trance":"pop",
    "ambient":"pop","new age":"pop","downtempo":"pop",
    "reggae":"pop","ska":"pop","dub":"pop",
    "country":"pop","folk":"pop","singer songwriter":"pop",
    "latin":"pop","salsa":"pop","cumbia":"pop","reggaeton":"pop",
    "k pop":"pop","j pop":"pop","city pop":"pop",
    "chanson":"pop","french pop":"pop","french rock":"rock",
}
ALIASES_KEYS: List[str] = sorted(ALIASES.keys(), key=len, reverse=True)

OVERRIDES: Dict[str, str] = {
    # "blues": "jazz",
}

def heuristic_language_prefix(s: str) -> Optional[str]:
    if s.startswith("french "):
        rest = s.replace("french ", "", 1)
        for k in ALIASES_KEYS:
            if k in rest:
                return ALIASES[k]
        return "pop"
    if "k pop" in s: return "pop"
    if "j pop" in s: return "pop"
    return None

def fuzzy_guess(s: str, cutoff: float = 0.86) -> Optional[str]:
    m = get_close_matches(s, ALIASES_KEYS, n=1, cutoff=cutoff)
    return ALIASES[m[0]] if m else None

def token_vote(s: str) -> Optional[str]:
    ts = tokens(s)
    votes: List[str] = []
    for k in ALIASES_KEYS:
        kk = k.split()
        if all(t in ts for t in kk):
            votes.append(ALIASES[k])
    if not votes:
        return None
    cnt = Counter(votes)
    top = max(cnt.values())
    cands = [g for g, c in cnt.items() if c == top]
    cands.sort(key=lambda g: PRIORITY.index(g))
    return cands[0]

def map_one_genre(genre_str: str) -> Dict[str, str]:
    raw = genre_str or ""
    s = norm(raw)

    if s in OVERRIDES:
        label = OVERRIDES[s]
        return {"raw": raw, "label": label, "id": str(COARSE[label]), "conf": "high", "how": "override"}

    h = heuristic_language_prefix(s)
    if h:
        return {"raw": raw, "label": h, "id": str(COARSE[h]), "conf": "high", "how": "lang-prefix"}

    for k in ALIASES_KEYS:
        if k in s:
            label = ALIASES[k]
            conf = "high" if len(k) >= 5 else "medium"
            return {"raw": raw, "label": label, "id": str(COARSE[label]), "conf": conf, "how": f"substr:{k}"}

    vote = token_vote(s)
    if vote:
        return {"raw": raw, "label": vote, "id": str(COARSE[vote]), "conf": "medium", "how": "token-vote"}

    guess = fuzzy_guess(s, cutoff=0.86)
    if guess:
        return {"raw": raw, "label": guess, "id": str(COARSE[guess]), "conf": "medium", "how": "fuzzy"}

    return {"raw": raw, "label": "pop", "id": str(COARSE["pop"]), "conf": "low", "how": "fallback"}

def parse_mbtags(cell) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)) or (isinstance(cell, str) and cell.strip()==""):
        return []
    s = str(cell).strip()
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    parts = re.split(r"[;,|,/]", s)
    return [p.strip() for p in parts if p.strip()]

def map_tags_to_coarse(mbtags: List[str]) -> Dict[str, str]:
    if not mbtags:
        return {"label": "pop", "id": str(COARSE["pop"]), "conf": "low", "how": "empty"}
    votes: List[str] = []
    hows: List[Tuple[str, str]] = []
    for t in mbtags:
        r = map_one_genre(t)
        votes.append(r["label"]); hows.append((t, r["how"]))
    cnt = Counter(votes)
    top = max(cnt.values())
    cands = [lbl for lbl, c in cnt.items() if c == top]
    cands.sort(key=lambda g: PRIORITY.index(g))
    chosen = cands[0]
    conf = "high" if top >= 2 or len(mbtags) == 1 else "medium"
    return {"label": chosen, "id": str(COARSE[chosen]), "conf": conf, "how": f"vote:{dict(cnt)}", "examples": str(hows[:3])}

def process_csv(path_in: str, path_out: str):
    df = pd.read_csv(path_in)
    required = {"artist","title","path","mbtags"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    results = []
    for mb in df["mbtags"].tolist():
        tags = parse_mbtags(mb)
        r = map_tags_to_coarse(tags)
        results.append(r)

    df["genre"] = [r["label"] for r in results]
    df["genre_id"] = [int(r["id"]) for r in results]
    #df["genre_conf"] = [r.get("conf","") for r in results]
    #df["genre_how"] = [r.get("how","") for r in results]

    df.to_csv(path_out, index=False)

    counts = df["genre"].value_counts().to_dict()
    print("Resumen por género:", counts)
    #low_conf = (df["genre_conf"]=="low").sum()
    #print(f"Filas con confianza 'low': {low_conf}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path_in", required=True, help="CSV de entrada")
    ap.add_argument("--out", dest="path_out", required=True, help="CSV de salida con columna 'genre'")
    args = ap.parse_args()
    process_csv(args.path_in, args.path_out)

if __name__ == "__main__":
    main()
