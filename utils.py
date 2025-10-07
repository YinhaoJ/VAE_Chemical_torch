import torch
import torch.nn.functional as F
import math
from rdkit import Chem

def vae_loss_function(recon_x, x, mu, log_var, pad_token_int, beta=1.0):
    """Calculates the VAE loss (Reconstruction + KL Divergence)."""
    recon_x_flat = recon_x.view(-1, recon_x.size(2))
    x_flat = x.reshape(-1)
    ce_loss = F.cross_entropy(recon_x_flat, x_flat, ignore_index=pad_token_int, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (ce_loss + beta * kld_loss) / x.size(0)

def kl_anneal_cyclical(epoch, total_epochs, cycles=4):
    """A cyclical annealing schedule for the KL divergence weight beta."""
    period = total_epochs / cycles
    beta = (epoch % period) / period # Ramps from 0 to 1 over each cycle
    return float(beta)

def decode_from_latents(model, z_batch, sos_token_int, eos_token_int, int_to_char, max_len, device):
    """Decodes a batch of latent vectors into SMILES strings."""
    model.eval()
    batch_size = z_batch.size(0)
    decoded_smiles_list = [""] * batch_size
    input_tensor = torch.tensor([[sos_token_int]] * batch_size, dtype=torch.long, device=device)
    hidden = torch.zeros(model.decoder.gru.num_layers, batch_size, model.decoder.gru.hidden_size, device=device)
    finished = [False] * batch_size

    with torch.no_grad():
        for _ in range(max_len):
            embedded = model.decoder.embedding(input_tensor)
            combined_input = torch.cat([embedded, z_batch.unsqueeze(1)], dim=2)
            output, hidden = model.decoder.gru(combined_input, hidden)
            output = model.decoder.fc_out(output)
            topi = output.argmax(2)

            for i in range(batch_size):
                if not finished[i]:
                    char_idx = topi[i].item()
                    if char_idx == eos_token_int:
                        finished[i] = True
                    else:
                        decoded_smiles_list[i] += int_to_char[char_idx]
            if all(finished): break
            input_tensor = topi
    return decoded_smiles_list

def generate_smiles(model, num_samples, latent_dim, sos_token_int, eos_token_int, int_to_char, max_len, device):
    """Generates new SMILES by sampling z and then calling the decoder."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated_smiles = decode_from_latents(model, z, sos_token_int,
                                               eos_token_int, int_to_char, max_len, device)
    return generated_smiles

def evaluate_metrics(generated_smiles, train_smiles_set):
    """Calculates validity, uniqueness, and novelty of generated molecules."""
    valid_mols = [mol for s in generated_smiles if (mol := Chem.MolFromSmiles(s)) is not None]
    if not valid_mols: return 0.0, 0.0, 0.0
    valid_smiles = [Chem.MolToSmiles(m) for m in valid_mols]
    
    validity = len(valid_smiles) / len(generated_smiles)
    uniqueness = len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0
    novelty = len(set(valid_smiles) - train_smiles_set) / len(set(valid_smiles)) if valid_smiles else 0
    
    return validity * 100, uniqueness * 100, novelty * 100
