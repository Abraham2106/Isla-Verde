import re
import unicodedata

import pandas as pd

INPUT_PATH = "data/Datos_Abiertos_ARESEP_Centrales_Eléctricas_.csv"
OUTPUT_PATH = "data/Datos_Abiertos_ARESEP_Centrales_Electricas_clean.csv"


TEXT_COLUMNS = [
    "operador",
    "central_electrica",
    "fuente_de_energia_electrica",
    "provincia",
    "canton",
    "distrito",
]


def normalize_column_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name


def normalize_text_value(value: str) -> str:
    if not isinstance(value, str):
        return value
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return value.strip().upper()


def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    df.columns = [normalize_column_name(c) for c in df.columns]
    for col in TEXT_COLUMNS:
        df[col] = df[col].apply(normalize_text_value)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Columnas normalizadas: {df.columns.tolist()}")
    print(f"Guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
