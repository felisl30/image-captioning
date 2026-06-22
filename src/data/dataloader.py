from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from src.data.dataset import MimicCXRDataset
from src.data.utils import load_json


def create_dataloader(
    hf_split,
    processor,
    indices_path: Optional[Path] = None,
    indices: Optional[list] = None,
    text_col="impression",
    batch_size=4,
    shuffle=True,
    num_workers=0,
    max_length=128
):
    """Crea un DataLoader para BLIP.

    Acepta índices como lista en memoria o como ruta a un JSON en disco.
    Exactamente uno de los dos debe ser proporcionado.

    Args:
        hf_split: split de HuggingFace Dataset.
        processor: BlipProcessor.
        indices_path: Path a un JSON con lista de índices. Alternativo a `indices`.
        indices: lista de índices ya cargada. Alternativa a `indices_path`.
        text_col: columna de texto target.
        batch_size: tamaño de batch.
        shuffle: si mezclar el dataset.
        num_workers: workers del DataLoader.
        max_length: longitud máxima de tokenización.
    """
    if indices is None and indices_path is not None:
        indices = load_json(indices_path)
    elif indices is None:
        raise ValueError("Se requiere exactamente uno de: indices o indices_path.")

    dataset = MimicCXRDataset(
        hf_split=hf_split,
        indices=indices,
        processor=processor,
        text_col=text_col,
        max_length=max_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return dataloader
