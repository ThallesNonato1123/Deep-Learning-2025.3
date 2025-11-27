from skimage.metrics import structural_similarity as ssim
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def compute_ssim_batch(outputs, images):
    total = 0
    batch = images.size(0)

    for i in range(batch):
        img_np = images[i].cpu().numpy().astype("float32").squeeze()
        out_np = outputs[i].cpu().numpy().astype("float32").squeeze()
        total += ssim(img_np, out_np, data_range=1.0)

    return total / batch

def save_metrics_plots(all_train_losses, all_val_losses, all_ssim, output_dir="metrics", modelType=""):
    os.makedirs(output_dir, exist_ok=True)
    mean_train = np.mean(np.array(all_train_losses), axis=0)
    mean_val = np.mean(np.array(all_val_losses), axis=0)
    mean_ssim = np.mean(np.array(all_ssim), axis=0)

    plt.figure(figsize=(10, 6))
    for i, fold_losses in enumerate(all_train_losses):
        plt.plot(np.log(fold_losses), label=f"Fold {i+1}")
    plt.title("Loss de Treino por Fold")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{modelType}-train_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, fold_losses in enumerate(all_val_losses):
        plt.plot(np.log(fold_losses), label=f"Fold {i+1}")
    plt.title("Loss de Validação por Fold")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{modelType}-val_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, fold_ssim in enumerate(all_ssim):
        plt.plot(fold_ssim, label=f"Fold {i+1}")
    plt.title("SSIM por Fold")
    plt.xlabel("Época")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{modelType}-ssim.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("Learning Curve - Média dos Folds")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.plot(np.log(mean_train), label="Treino (média)")
    plt.plot(np.log(mean_val), label="Validação (média)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{modelType}-learningCurveFolds.png"))
    plt.close()

    print(f"Gráficos salvos em: {output_dir}/")

def show_autoencoder_results(model, loader, device='cuda', num_images=8,
                             output_dir="outputs", filename="autoencoder_output.png"):
    model.eval()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(base_dir, ".."))
    output_dir = os.path.join(parent_dir, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

    images, _ = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        x = images.view(-1, 28*28)
        z = model.encoder(x)
        outputs = model.decoder(z)

    outputs = outputs.view(-1, 1, 28, 28).cpu()
    images = images.cpu()

    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(3, num_images, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title("Input")
        plt.axis("off")

        plt.subplot(3, num_images, 2*num_images + i + 1)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        plt.title("Decoded")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Output salvo em: {save_path}")

def show_vae_results(model, loader, device='cuda', num_images=8,
                     output_dir="outputs", filename="vae_output.png"):
    model.eval()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(base_dir, ".."))
    output_dir = os.path.join(parent_dir, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

    images, _ = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        x = images
        outputs, mu, logvar = model(x)

    outputs = outputs.cpu()
    images = images.cpu()

    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title("Input")
        plt.axis("off")

        plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(outputs[i].squeeze(), cmap="gray")
        plt.title("Decoded")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"VAE output salvo em: {save_path}")
