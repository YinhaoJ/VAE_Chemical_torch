import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data_loader import get_data_and_vocab, SmilesDataset, collate_fn
from model import ConvEncoder, GruDecoder, VAE
from utils import vae_loss_function, kl_anneal_cyclical, generate_smiles, evaluate_metrics

def main():
    # --- Hyperparameters ---
    MAX_LEN = 120
    EMBEDDING_DIM = 128
    LATENT_DIM = 196
    DECODER_HIDDEN_DIM = 488
    DECODER_NUM_LAYERS = 3
    BATCH_SIZE = 128
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    SAVE_PATH = 'chemical_vae.pth'

    # --- 1. Data Loading and Setup ---
    smiles_list, charset, char_to_int, int_to_char = get_data_and_vocab()
    vocab_size = len(charset)
    pad_token_int = char_to_int['<pad>']
    sos_token_int = char_to_int['<sos>']
    eos_token_int = char_to_int['<eos>']

    dataset = SmilesDataset(smiles_list, char_to_int)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                             collate_fn=lambda b: collate_fn(b, pad_token_int))

    # --- 2. Model Initialization ---
    print("\n-> Initializing Model and Optimizer")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    encoder = ConvEncoder(vocab_size, EMBEDDING_DIM, LATENT_DIM, pad_token_int).to(device)
    decoder = GruDecoder(vocab_size, EMBEDDING_DIM, LATENT_DIM, DECODER_HIDDEN_DIM, DECODER_NUM_LAYERS).to(device)
    model = VAE(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training ---
    print("\n-> Starting Training Loop")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        beta = kl_anneal_cyclical(epoch, EPOCHS)

        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}, Beta: {beta:.3f}"):
            sequences = batch.to(device)
            optimizer.zero_grad()

            input_seq = sequences[:, :-1]
            target_seq = sequences[:, 1:]

            recon_sequences, mu, log_var = model(input_seq)
            loss = vae_loss_function(recon_sequences, target_seq, mu, log_var, pad_token_int, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

    print("Training complete!")
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

    # --- 4. Generation and Evaluation ---
    print("\n-> Generating and Evaluating New Molecules")
    train_smiles_set = set(smiles_list)
    generated_smiles = generate_smiles(model, 1000, LATENT_DIM, sos_token_int, eos_token_int, int_to_char, MAX_LEN, device)
    validity, uniqueness, novelty = evaluate_metrics(generated_smiles, train_smiles_set)

    print("\nEvaluation Metrics on 1000 generated molecules:")
    print(f"  Validity: {validity:.2f}%")
    print(f"  Uniqueness: {uniqueness:.2f}%")
    print(f"  Novelty: {novelty:.2f}%")

if __name__ == '__main__':
    main()
