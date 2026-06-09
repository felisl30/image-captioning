from torch.utils.data import Dataset
from src.data.utils import validate_columns, get_sample


class MimicCXRDataset(Dataset):
    """
    Dataset de PyTorch para MIMIC-CXR.

    Funciona tanto con:
    - Dataset de Hugging Face
    - lista de diccionarios creada con streaming/take()

    Devuelve un item compatible con BLIP:
    - pixel_values
    - input_ids
    - attention_mask
    - labels
    - idx
    - text
    """

    def __init__(
        self,
        hf_split,
        indices,
        processor,
        text_col="impression",
        max_length=128
    ):
        self.hf_split = hf_split
        self.indices = indices
        self.processor = processor
        self.text_col = text_col
        self.max_length = max_length

        validate_columns(
            hf_split,
            expected_columns=["image", text_col]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = get_sample(self.hf_split, real_idx)

        image = sample["image"].convert("RGB")
        text = sample[self.text_col]

        encoding = self.processor(
            images=image,
            text=text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {}

        for key, value in encoding.items():
            item[key] = value.squeeze(0)

        labels = item["input_ids"].clone()

        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        item["labels"] = labels
        item["idx"] = real_idx
        item["text"] = text

        return item
