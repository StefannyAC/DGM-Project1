# homologar_generos.py (compat Python 3.8+)
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda con asistencia de ChatGPT
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Script para homologar géneros musicales a 4 clases: classical, jazz, rock, pop
# Basado en etiquetas de MusicBrainz y heurísticas.
# Usarlo solo si se necesita procesar un CSV con etiquetas de géneros. Para el proyecto se dejan los csv ya procesados.
# ============================================================
import re
import json
import argparse
import unicodedata
from difflib import get_close_matches
from collections import Counter
from typing import Optional, List, Dict, Tuple

import pandas as pd

COARSE: Dict[str, int] = {"classical": 0, "jazz": 1, "rock": 2, "pop": 3} # Clases coarse
PRIORITY: List[str] = ["classical", "jazz", "rock", "pop"] # Prioridad en caso de empate

def norm(s: str) -> str:
    """
    Función que normaliza cadena de texto:
    - quita acentos
    - minúsculas
    - normaliza separadores y puntuación
    - quita espacios extras

    Args:
        s: cadena de texto
    Returns:
        cadena normalizada
    """
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii") # quitar acentos
    s = s.lower() # minúsculas
    s = s.replace("&", " and ") # normalizar &
    s = re.sub(r"[-_/]+", " ", s) # normalizar separadores
    s = re.sub(r"[^\w\s]+", " ", s) # quitar puntuación
    s = re.sub(r"\s+", " ", s).strip() # espacios extras
    return s

def tokens(s: str) -> List[str]:
    """ 
    Función que tokeniza una cadena normalizada 
    
    Args:
        s: cadena normalizada
    Returns:
        lista de tokens
    """
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
ALIASES_KEYS: List[str] = sorted(ALIASES.keys(), key=len, reverse=True) # claves ordenadas por longitud (largo a corto)

# En caso de que se quieran hacer overrides manuales
OVERRIDES: Dict[str, str] = {
    # "blues": "jazz", 
}

def heuristic_language_prefix(s: str) -> Optional[str]:
    """
    Función que detecta prefijos de idioma para asignar género.
    
    Args:
        s: cadena normalizada
    Returns:
        género detectado o None
    """
    if s.startswith("french "): # si empieza con "french "
        rest = s.replace("french ", "", 1) # quitar prefijo
        for k in ALIASES_KEYS: # buscar en el resto
            if k in rest: # si coincide
                return ALIASES[k] # devolver género
        return "pop" # si no coincide, asignar pop
    if "k pop" in s: return "pop" # k pop
    if "j pop" in s: return "pop" # j pop
    return None # no detectado

def fuzzy_guess(s: str, cutoff: float = 0.86) -> Optional[str]:
    """
    Función que usa coincidencia difusa para adivinar género. Esto es, asigna el género cuyo alias
    es más parecido a la cadena dada, si la similitud es mayor que el cutoff.
    """
    m = get_close_matches(s, ALIASES_KEYS, n=1, cutoff=cutoff) # obtener mejor coincidencia
    return ALIASES[m[0]] if m else None # devolver género o None

def token_vote(s: str) -> Optional[str]:
    """
    Función que vota por tokens en la cadena para asignar género.

    Args:
        s: cadena normalizada
    Returns:
        género votado o None
    """
    ts = tokens(s) # tokenizar
    votes: List[str] = [] # lista de votos
    for k in ALIASES_KEYS: # para cada alias
        kk = k.split() # tokens del alias
        if all(t in ts for t in kk): # si todos los tokens están en la cadena
            votes.append(ALIASES[k]) # votar por el género
    if not votes: # no hay votos
        return None # devolver None
    cnt = Counter(votes)  # contar votos
    top = max(cnt.values()) # votos máximos
    cands = [g for g, c in cnt.items() if c == top] # candidatos con más votos
    cands.sort(key=lambda g: PRIORITY.index(g)) # ordenar por prioridad
    return cands[0] # devolver el de mayor prioridad

def map_one_genre(genre_str: str) -> Dict[str, str]: 
    """ 
    Función que mapea una cadena de género a una clase coarse.
    Args:
        genre_str: cadena de género
    Returns:
        diccionario con keys: raw, label, id, conf, how
    """
    raw = genre_str or "" # manejar None
    s = norm(raw) # normalizar

    if s in OVERRIDES:
        label = OVERRIDES[s] # usar override
        return {"raw": raw, "label": label, "id": str(COARSE[label]), "conf": "high", "how": "override"} # devolver resultado

    h = heuristic_language_prefix(s) # usar heurística de prefijo de idioma
    if h:
        return {"raw": raw, "label": h, "id": str(COARSE[h]), "conf": "high", "how": "lang-prefix"} # devolver resultado

    for k in ALIASES_KEYS:
        if k in s:
            label = ALIASES[k] # obtener género
            conf = "high" if len(k) >= 5 else "medium" # confianza según longitud
            return {"raw": raw, "label": label, "id": str(COARSE[label]), "conf": conf, "how": f"substr:{k}"} # devolver resultado

    vote = token_vote(s) # usar votación por tokens
    if vote:
        return {"raw": raw, "label": vote, "id": str(COARSE[vote]), "conf": "medium", "how": "token-vote"} # devolver resultado

    guess = fuzzy_guess(s, cutoff=0.86) # usar coincidencia difusa
    if guess:
        return {"raw": raw, "label": guess, "id": str(COARSE[guess]), "conf": "medium", "how": "fuzzy"} # devolver resultado

    return {"raw": raw, "label": "pop", "id": str(COARSE["pop"]), "conf": "low", "how": "fallback"} # fallback a pop

def parse_mbtags(cell) -> List[str]:
    """
    Función que parsea la celda mbtags del CSV.
    Puede ser JSON, o una cadena separada por varios delimitadores.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)) or (isinstance(cell, str) and cell.strip()==""): # si está vacío
        return [] # devolver lista vacía
    s = str(cell).strip() # convertir a cadena
    try:
        data = json.loads(s) # intentar parsear JSON
        if isinstance(data, list): # si es lista 
            return [str(x) for x in data] # devolver lista de strings
    except Exception:
        pass
    parts = re.split(r"[;,|,/]", s) # separar por varios delimitadores
    return [p.strip() for p in parts if p.strip()] # devolver lista limpia

def map_tags_to_coarse(mbtags: List[str]) -> Dict[str, str]:
    """
    Función que mapea una lista de etiquetas de MusicBrainz a una clase coarse.
    Args:
        mbtags: Lista de etiquetas MusicBrainz (strings) para una pista.

    Returns:
        Dict: Diccionario con claves:
            - "label": género coarse elegido (str).
            - "id": id (string) del género coarse ('COARSE[label]' convertido a str).
            - "conf": "high" | "medium" | "low" (criterio descrito arriba).
            - "how": explicación compacta (p.ej., conteo de votos).
            - "examples": muestra de (tag_original, cómo_se_mapeó) para depuración.
    """
    if not mbtags:
        return {"label": "pop", "id": str(COARSE["pop"]), "conf": "low", "how": "empty"} # si está vacío, asignar pop
    votes: List[str] = []  # Acumulará las etiquetas coarse mapeadas (uno por tag).
    hows: List[Tuple[str, str]] = [] # Guarda (tag_original, "how") para auditoría.

    # Mapea cada tag MusicBrainz a una etiqueta coarse con explicación.
    for t in mbtags:
        r = map_one_genre(t) # Esperado: {"label": <coarse>, "how": <detalle_del_mapeo>}
        votes.append(r["label"]); hows.append((t, r["how"])) # Suma un voto para la etiqueta coarse asignada y traza como se decidió.
    cnt = Counter(votes) # Conteo de votos por etiqueta coarse.
    top = max(cnt.values()) # Máximo número de votos alcanzado por alguna clase.
    cands = [lbl for lbl, c in cnt.items() if c == top] # Lista de candidatos empatados con el máximo número de votos.
    # Desempate por prioridad explícita: el que aparezca primero en PRIORITY gana.
    cands.sort(key=lambda g: PRIORITY.index(g)) 
    chosen = cands[0] # Etiqueta coarse elegida.
    # Heurística de confianza: alta si el ganador tiene >=2 votos o si solo había 1 tag.
    conf = "high" if top >= 2 or len(mbtags) == 1 else "medium"
    # Respuesta con algunos ejemplos de cómo se mapeó (solo primeros 3 para no saturar).
    return {"label": chosen, "id": str(COARSE[chosen]), "conf": conf, "how": f"vote:{dict(cnt)}", "examples": str(hows[:3])}

def process_csv(path_in: str, path_out: str):
    """
    Función para procesar un CSV con columnas 'artist', 'title', 'path', 'mbtags',
    generar una columna 'genre' (coarse) y 'genre_id', y escribir un nuevo CSV.

    Args:
        path_in: Ruta al CSV de entrada con 'mbtags' sin procesar.
        path_out: Ruta del CSV de salida con columnas 'genre' y 'genre_id'.

    Raises:
        ValueError: si faltan columnas requeridas.
    """
    # Carga el CSV de entrada.
    df = pd.read_csv(path_in)
    required = {"artist","title","path","mbtags"} # Conjunto mínimo de columnas esperadas.
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    results = [] # Guardará los dicts devueltos por map_tags_to_coarse por cada fila.
    # Aplica el mapeo fila a fila: parsea mbtags (string) y luego decide coarse.
    for mb in df["mbtags"].tolist():
        tags = parse_mbtags(mb)
        r = map_tags_to_coarse(tags)
        results.append(r)

    # Extrae columnas de interés del resultado y las adjunta al DataFrame.
    df["genre"] = [r["label"] for r in results]
    df["genre_id"] = [int(r["id"]) for r in results]

    # Escribe el CSV de salida.
    df.to_csv(path_out, index=False)

    # Resumen por consola para ver distribución de clases.
    counts = df["genre"].value_counts().to_dict()
    print("Resumen por género:", counts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path_in", required=True, help="CSV de entrada")
    ap.add_argument("--out", dest="path_out", required=True, help="CSV de salida con columna 'genre'")
    args = ap.parse_args()
    process_csv(args.path_in, args.path_out)

if __name__ == "__main__":
    main()