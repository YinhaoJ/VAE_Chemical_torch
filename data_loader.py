import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import requests
import os
from tqdm import tqdm

class SmilesDataset(Dataset):
    """
    Custom PyTorch Dataset for handling SMILES strings.
    - Converts SMILES to tokenized integer tensors.
    - Adds <sos> and <eos> tokens.
    """
    def __init__(self, smiles_list, char_to_int):
        self.smiles_list = smiles_list
        self.char_to_int = char_to_int

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        tokens = ['<sos>'] + list(smiles) + ['<eos>']
        encoded = [self.char_to_int[char] for char in tokens]
        return torch.tensor(encoded, dtype=torch.long)

def collate_fn(batch, pad_token_int):
    """
    Custom collate function for the DataLoader.
    - Pads all sequences in a batch to the same length.
    """
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=pad_token_int)
    return padded_batch

def get_data_and_vocab():
    """
    Downloads the ZINC dataset if not found locally, then creates the character vocabulary.
    
    Returns:
        tuple: A tuple containing:
            - smiles_list (list): List of all SMILES strings.
            - charset (list): The complete vocabulary of characters.
            - char_to_int (dict): Mapping from character to integer.
            - int_to_char (dict): Mapping from integer to character.
    """
    print("-> Loading and Preprocessing Data")
    
    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    filename = "250k_rndm_zinc_drugs_clean_3.csv"

    if not os.path.exists(filename):
        print(f"Dataset not found locally. Downloading from {url}...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size_in_bytes,
            unit='iB',
            unit_scale=True,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
             print("ERROR, something went wrong during download")
        print("Download complete.")
   
    df = pd.read_csv(filename)
    df["smiles"] = df["smiles"].apply(lambda s: s.replace("\n", ""))
    smiles_list = df['smiles'].tolist()

    print("-> Building vocabulary...")
    all_chars = set(''.join(smiles_list))
    special_tokens = ['<pad>', '<sos>', '<eos>']
    charset = special_tokens + sorted(list(all_chars))

    char_to_int = {c: i for i, c in enumerate(charset)}
    int_to_char = {i: c for c, i in char_to_int.items()}
    
    print(f"Vocabulary size: {len(charset)}")
    return smiles_list, charset, char_to_int, int_to_char
