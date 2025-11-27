# generate.py

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms

from evaluate import show_autoencoder_results
from model import Autoencoder, VAE


def main():
    parser = argparse.ArgumentParser(description="Gerar imagens usando um modelo salvo (.pth)")
    parser.add_argument("--model-path", type=str, required=True, help="Caminho do arquivo .pth")
    parser.add_argument("--model-type", type=str, choices=["ae", "vae"], required=True)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--filename", type=str, default="generated.png")

    args = parser.parse_args()

    # Seleção de device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📸 Gerando imagens com {args.model_path} — device: {device}\n")

    # ------------------------------
    # Carregar modelo
    # ------------------------------
    if args.model_type == "ae":
        model = Autoencoder().to(device)
    else:
        model = VAE(args.latent_dim).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # ------------------------------
    # Dataset FashionMNIST
    # ------------------------------
    transform = transforms.ToTensor()
    dataset = FashionMNIST("../data", download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ------------------------------
    # Gerar imagens
    # ------------------------------
    show_autoencoder_results(
        model=model,
        loader=loader,
        device=device,
        num_images=args.num_images,
        output_dir=args.output_dir,
        filename=args.filename
    )


if __name__ == "__main__":
    main()
