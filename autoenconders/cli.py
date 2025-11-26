import argparse
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from .train import train_autoenconder_model, train_variational_autoenconder_model_kfold
from .test import test_model, test_model_vae


def main():
    parser = argparse.ArgumentParser(description="Treinar Autoencoder com K-Fold")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    # ADICIONADO:
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["ae", "vae"],
        default="ae",
        help="Escolha entre ae (autoencoder) ou vae (variational autoencoder)"
    )

    args = parser.parse_args()

    # Dataset
    transform = transforms.ToTensor()
    dataset = FashionMNIST("./data", download=True, transform=transform)

    # Train/test split
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        random_state=42
    )
    train_dataset = Subset(dataset, train_idx)
    test_dataset  = Subset(dataset, test_idx)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seleção de modelo
    if args.model_type == "ae":
        print("\n🔵 Usando Autoencoder normal\n")
        model = train_autoenconder_model(
            dataset=train_dataset,
            k=args.k,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

    elif args.model_type == "vae":
        print("\n🟣 Usando Variational Autoencoder (VAE)\n")
        model = train_variational_autoenconder_model_kfold(
            dataset=train_dataset,
            k=args.k,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

    model = model.to(device)

    # Teste
    if args.model_type == "ae":
        test_model(
            model=model,
            test_dataset=test_dataset,
            device=device
        )
    else:
        test_model_vae(
            model=model,
            test_dataset=test_dataset,
            device=device
        )
    
