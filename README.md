# Chemical VAE in PyTorch

This project is a PyTorch implementation of a Variational Autoencoder (VAE) for automatic chemical design, based on the methodology from the paper ["Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules"](https://pubs.acs.org/doi/10.1021/acscentsci.7b00572) by Gómez-Bombarelli et al.

The model learns a continuous latent space representation of molecules from their SMILES strings (Here using ZINC data). By sampling from this latent space, we can generate novel molecules with desired properties.

## Architecture

* **Encoder**: A series of 1D Convolutional Neural Networks (CNNs) are used to process the input SMILES string and encode it into a latent vector ($z$). Global Max Pooling handles variable-length inputs.
* **Decoder**: A Gated Recurrent Unit (GRU) based network decodes a vector from the latent space back into a SMILES string. The latent vector is fed as an input at each time step of the decoding process.



## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/YinhaoJ/VAE_Chemical_torch.git
    cd VAE_Chemical_torch
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training a New Model

To train the VAE from scratch on the ZINC dataset, run the main training script:

```bash
python train.py
```

The script will download the dataset, preprocess it, train the model, save the weights to `chemical_vae.pth`, and print evaluation metrics for a sample of generated molecules.

### Generating Molecules from a Trained Model

To generate new molecules using the pre-trained weights, you can use the `generate.py` script:

```bash
python generate.py --weights_path chemical_vae.pth --num_samples 50
```

This will load the model, sample 50 points from the latent space, decode them into SMILES strings, and print the valid ones.

## Acknowledgments and Attribution

This project is an implementation of the chemical VAE methodology introduced in the following paper. The core architecture and concepts are derived from this work:

- **Gómez-Bombarelli, R., et al. (2018).** *Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules.* ACS Central Science, 4(2), 268-276. [DOI: 10.1021/acscentsci.7b00572](https://doi.org/10.1021/acscentsci.7b00572)

The specific dataset used for training is a curated 250,000-molecule subset of the **ZINC database**, which was provided by the paper's authors in their original [GitHub repository](https://github.com/aspuru-guzik-group/chemical_vae).