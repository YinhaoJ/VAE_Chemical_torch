import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import argparse

from data_loader import get_data_and_vocab
from model import ConvEncoder, GruDecoder, VAE
from utils import generate_smiles

def main(args):
    # --- 1. Hyperparameters and Data Loading ---
    # These must match the settings used during training
    MAX_LEN = 120
    EMBEDDING_DIM = 128
    LATENT_DIM = 196
    DECODER_HIDDEN_DIM = 488
    DECODER_NUM_LAYERS = 3
    
    # Load the vocabulary from the original dataset
    _, _, char_to_int, int_to_char = get_data_and_vocab()
    vocab_size = len(char_to_int)
    pad_token_int = char_to_int['<pad>']
    sos_token_int = char_to_int['<sos>']
    eos_token_int = char_to_int['<eos>']

    # --- 2. Instantiate and Load the Model ---
    print("-> Instantiating the model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    encoder = ConvEncoder(vocab_size, EMBEDDING_DIM, LATENT_DIM, pad_token_int).to(device)
    decoder = GruDecoder(vocab_size, EMBEDDING_DIM, LATENT_DIM, DECODER_HIDDEN_DIM, DECODER_NUM_LAYERS).to(device)
    model = VAE(encoder, decoder).to(device)

    print(f"-> Loading saved weights from {args.weights_path}...")
    try:
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Weights file not found at {args.weights_path}.")
        print("Please run train.py first to generate the model weights.")
        return

    model.eval()
    print("Model loaded successfully!")

    # --- 3. Generate a large batch of molecules ---
    print(f"\n-> Generating {args.num_samples} molecules...")
    generated_smiles = generate_smiles(model, args.num_samples, LATENT_DIM, sos_token_int, 
                                       eos_token_int, int_to_char, MAX_LEN, device)
    
    # --- 4. Filter them based on properties ---
    print("-> Filtering generated molecules by chemical properties...")
    valid_smiles = [s for s in generated_smiles if Chem.MolFromSmiles(s) is not None]
    
    results = []
    for smiles in valid_smiles:
        mol = Chem.MolFromSmiles(smiles)
        qed = Descriptors.qed(mol)
        mol_wt = Descriptors.MolWt(mol)
        log_p = Descriptors.MolLogP(mol)

        # Filtering criteria: High drug-likeness (QED) and reasonable molecular weight
        if qed > 0.7 and 200 < mol_wt < 500:
            results.append({
                "smiles": smiles,
                "qed": qed,
                "mol_wt": mol_wt,
                "log_p": log_p
            })
    
    if not results:
        print("\nNo molecules met the filtering criteria. Try generating a larger sample.")
        return

    # --- 5. Display the best candidates ---
    results_df = pd.DataFrame(results)
    print("\n--- Top 10 Generated Molecules Meeting Criteria (sorted by QED) ---")
    print(results_df.sort_values(by='qed', ascending=False).head(10).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and filter molecules using a trained Chemical VAE.")
    parser.add_argument('--weights_path', type=str, default='chemical_vae.pth', help='Path to the trained model weights file.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of molecules to generate.')
    
    args = parser.parse_args()
    main(args)
