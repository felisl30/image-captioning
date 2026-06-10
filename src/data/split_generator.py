import random
from src.data.utils import save_json, get_valid_indices


def compute_split_sizes(
    n_valid,
    train_size=24000,
    val_size=4500,
    test_size=1500,
    selected_size=30,
    auto_shrink=True,
    train_ratio=0.80,
    val_ratio=0.15
):
    """
    Calcula tamaños de split.

    Si hay suficientes datos, usa los tamaños pedidos.
    Si no hay suficientes datos y auto_shrink=True, usa proporciones.
    """

    needed = train_size + val_size + test_size

    if n_valid >= needed:
        return train_size, val_size, test_size, min(selected_size, test_size)

    if not auto_shrink:
        raise ValueError(
            f"No hay suficientes muestras válidas. "
            f"Disponibles: {n_valid}, necesarias: {needed}"
        )

    new_train_size = int(train_ratio * n_valid)
    new_val_size = int(val_ratio * n_valid)
    new_test_size = n_valid - new_train_size - new_val_size
    new_selected_size = min(selected_size, new_test_size)

    if new_train_size <= 0 or new_val_size <= 0 or new_test_size <= 0:
        raise ValueError(
            f"No hay suficientes muestras válidas para generar splits. "
            f"Muestras válidas: {n_valid}"
        )

    return new_train_size, new_val_size, new_test_size, new_selected_size


def generate_splits(
    hf_split,
    text_col="impression",
    train_size=24000,
    val_size=4500,
    test_size=1500,
    selected_size=30,
    seed=42,
    output_dir="data/splits",
    selected_output="data/selected_indices.json",
    auto_shrink=True
):
    """
    Genera splits reproducibles usando índices.

    Funciona tanto con:
    - Dataset de Hugging Face
    - lista de diccionarios creada con streaming/take()

    Archivos generados:
    - data/splits/train_indices.json
    - data/splits/val_indices.json
    - data/splits/test_indices.json
    - data/selected_indices.json
    """

    valid_indices = get_valid_indices(hf_split, text_col=text_col)

    n_valid = len(valid_indices)

    train_size, val_size, test_size, selected_size = compute_split_sizes(
        n_valid=n_valid,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        selected_size=selected_size,
        auto_shrink=auto_shrink
    )

    rng = random.Random(seed)
    rng.shuffle(valid_indices)

    train_indices = valid_indices[:train_size]

    val_start = train_size
    val_end = train_size + val_size
    val_indices = valid_indices[val_start:val_end]

    test_start = val_end
    test_end = val_end + test_size
    test_indices = valid_indices[test_start:test_end]

    selected_indices = test_indices[:selected_size]

    save_json(train_indices, f"{output_dir}/train_indices.json")
    save_json(val_indices, f"{output_dir}/val_indices.json")
    save_json(test_indices, f"{output_dir}/test_indices.json")
    save_json(selected_indices, selected_output)

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
        "selected": selected_indices,
        "n_valid": n_valid,
        "used_train_size": train_size,
        "used_val_size": val_size,
        "used_test_size": test_size,
        "used_selected_size": selected_size,
    }
