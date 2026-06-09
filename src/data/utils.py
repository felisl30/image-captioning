import json
from pathlib import Path
from datasets import load_dataset


def load_mimic_dataset(cache_dir=None):
    """
    Carga el dataset MIMIC-CXR desde Hugging Face.
    """
    return load_dataset(
        "itsanmolgupta/mimic-cxr-dataset",
        cache_dir=cache_dir
    )


def is_empty_text(x):
    """
    Devuelve True si el texto es None o está vacío.
    """
    return x is None or str(x).strip() == ""


def save_json(obj, path):
    """
    Guarda un objeto Python como JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    """
    Carga un archivo JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_column_names(data):
    """
    Devuelve las columnas tanto si data es:
    - Dataset de Hugging Face
    - lista de diccionarios
    """

    if hasattr(data, "column_names"):
        return list(data.column_names)

    if isinstance(data, list):
        if len(data) == 0:
            return []

        if not isinstance(data[0], dict):
            raise TypeError(
                "Si data es una lista, se esperaba una lista de diccionarios."
            )

        return list(data[0].keys())

    raise TypeError(
        f"Tipo de data no soportado: {type(data)}. "
        "Se esperaba un Dataset de Hugging Face o una lista de diccionarios."
    )


def validate_columns(data, expected_columns):
    """
    Verifica que existan las columnas esperadas.
    Funciona para Dataset de Hugging Face y listas de diccionarios.
    """

    actual_columns = get_column_names(data)

    for col in expected_columns:
        if col not in actual_columns:
            raise ValueError(
                f"Falta la columna esperada: {col}. "
                f"Columnas disponibles: {actual_columns}"
            )

    return actual_columns


def get_sample(data, idx):
    """
    Devuelve una muestra por índice.
    Funciona para Dataset de Hugging Face y listas.
    """

    return data[idx]


def get_texts(data, text_col="impression"):
    """
    Devuelve la columna de texto como lista.
    Funciona para Dataset de Hugging Face y listas de diccionarios.
    """

    validate_columns(data, [text_col])

    if hasattr(data, "column_names"):
        return data[text_col]

    if isinstance(data, list):
        return [sample[text_col] for sample in data]

    raise TypeError(
        f"Tipo de data no soportado: {type(data)}."
    )


def get_valid_indices(data, text_col="impression"):
    """
    Devuelve índices donde la columna de texto elegida no esté vacía.
    Funciona para Dataset de Hugging Face y listas de diccionarios.
    """

    texts = get_texts(data, text_col=text_col)

    return [
        i for i, txt in enumerate(texts)
        if not is_empty_text(txt)
    ]
