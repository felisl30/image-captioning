import torch
from torch.utils.data import DataLoader
from src.data.dataset import MimicCXRDataset
from src.data.utils import load_json


def create_dataloader(
    hf_split,
    indices_path,
    processor,
    text_col="impression",
    batch_size=4,
    shuffle=True,
    num_workers=0,
    max_length=128
):
    """
    Crea un DataLoader para BLIP usando índices guardados en JSON.
    """

    indices = load_json(indices_path)

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
