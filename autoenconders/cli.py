# cli.py
import argparse
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from .train import train_autoenconder_model, train_variational_autoenconder_model_kfold
from .test import test_model, test_model_vae
from .evaluate import show_autoencoder_results, show_vae_results
from .model import Autoencoder, VAE


def build_parser():
    parser = argparse.ArgumentParser(description="CLI para Autoencoder/VAE com train e generate")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -----------------------------
    # 🔵 Subcomando: TRAIN
    # -----------------------------
    train_p = subparsers.add_parser("train", help="Treina AE ou VAE")
    train_p.add_argument("--k", type=int, default=5)
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--latent-dim", type=int, default=2)
    train_p.add_argument("--model-type", type=str, choices=["ae", "vae"], default="ae")
    train_p.add_argument("--save-path", type=str, default="model.pth")

    # -----------------------------
    # 🖼️ Subcomando: GENERATE
    # -----------------------------
    gen_p = subparsers.add_parser("generate", help="Gera imagens usando modelo salvo")
    gen_p.add_argument("--model-path", type=str, required=True)
    gen_p.add_argument("--model-type", type=str, choices=["ae", "vae"], required=True)
    gen_p.add_argument("--latent-dim", type=int, default=2)
    gen_p.add_argument("--num-images", type=int, default=8)
    gen_p.add_argument("--output-dir", type=str, default="outputs")
    gen_p.add_argument("--filename", type=str, default="output.png")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------
    if args.command == "train":
        print("\n🚀 Entrando no modo treino...\n")

        transform = transforms.ToTensor()
        dataset = FashionMNIST("./data", download=True, transform=transform)

        train_idx, test_idx = train_test_split(
            list(range(len(dataset))),
            test_size=0.2,
            random_state=42
        )
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        if args.model_type == "ae":
            model = train_autoenconder_model(
                train_dataset,
                k=args.k,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device
            )
            test_model(model, test_dataset, device)

        else:
            model = train_variational_autoenconder_model_kfold(
                train_dataset,
                k=args.k,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                latent_dim=args.latent_dim,
                device=device
            )
            test_model_vae(model, test_dataset, device)

        torch.save(model.state_dict(), args.save_path)
        print(f"💾 Modelo salvo em {args.save_path}")
        return

    # -------------------------------------------------------------------
    # GENERATE
    # -------------------------------------------------------------------
    if args.command == "generate":
        print("\n📸 Gerando imagens...\n")

        # Carregar modelo
        if args.model_type == "ae":
            model = Autoencoder().to(device)
        else:
            model = VAE(args.latent_dim).to(device)

        model.load_state_dict(torch.load(args.model_path, map_location=device))

        # Dataset para obter imagens reais
        transform = transforms.ToTensor()
        dataset = FashionMNIST("./data", download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        if args.model_type == "ae":
            show_autoencoder_results(
                model=model,
                loader=loader,
                device=device,
                num_images=args.num_images,
                output_dir=args.output_dir,
                filename=args.filename
            )
        else:
            show_vae_results(
                model=model,
                loader=loader,
                device=device,
                num_images=args.num_images,
                output_dir=args.output_dir,
                filename=args.filename
            )