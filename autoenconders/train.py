import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

from .model import Autoencoder
from .model import VAE
from .evaluate import compute_ssim_batch, save_metrics_plots 

def train_autoenconder_model(
    dataset,
    k=5,
    num_epochs=50,
    batch_size=64,
    lr=1e-3,
    device="cpu"
):
    criterion = nn.MSELoss()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # listas para métricas K-Fold
    all_train_losses = []
    all_val_losses = []
    all_ssim = []

    print("========== TREINO K-FOLD PARA AVALIAÇÃO ==========")
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n=========== FOLD {fold+1}/{k} ===========")

        # Subsets
        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)

        training_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Modelo do fold
        model = Autoencoder().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        # Histórico
        train_losses = []
        val_losses = []
        ssim_list = []

        for epoch in range(1, num_epochs + 1):
            # ---------- TREINO ----------
            model.train()
            running_loss = 0
            train_bar = tqdm(training_loader, desc=f"🔵 Treinando Fold {fold+1} Epoch {epoch}", leave=False)
            for images, _ in train_bar:
                images = images.to(device)
                flat = images.reshape(images.size(0), -1)

                optimizer.zero_grad()
                outputs = model(flat)
                loss = criterion(outputs, flat)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running_loss / len(training_loader)
            train_losses.append(train_loss)

            # ---------- VALIDAÇÃO ----------
            model.eval()
            running_val_loss = 0
            running_val_ssim = 0
            val_bar = tqdm(validation_loader, desc=f"🟣 Validando Fold {fold+1} Epoch {epoch}", leave=False)
            with torch.no_grad():
                for images, _ in val_bar:
                    images = images.to(device)
                    flat = images.reshape(images.size(0), -1)
                    outputs = model(flat).reshape(-1, 1, 28, 28)

                    # MSE
                    loss = criterion(outputs.reshape(images.size(0), -1), flat)
                    running_val_loss += loss.item()

                    # SSIM
                    batch_ssim = compute_ssim_batch(outputs, images)
                    running_val_ssim += batch_ssim

                    val_bar.set_postfix(val_loss=f"{loss.item():.4f}", ssim=f"{batch_ssim:.4f}")

            val_loss = running_val_loss / len(validation_loader)
            mean_ssim = running_val_ssim / len(validation_loader)

            val_losses.append(val_loss)
            ssim_list.append(mean_ssim)

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_ssim.append(ssim_list)

    print("\nK-Fold concluído!")
    torch.save(model.state_dict(), f".ae_model_fold_{fold+1}.pth")
    save_metrics_plots(all_train_losses, all_val_losses, all_ssim, modelType = "ae")

    print("\n========== TREINANDO MODELO FINAL EM TODO O DATASET DE TREINO ==========")
    final_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    final_model = Autoencoder().to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(1, num_epochs + 1):
        final_model.train()
        running_loss = 0
        final_bar = tqdm(final_loader, desc=f"🔵 Treinando Modelo Final Epoch {epoch}", leave=False)
        for images, _ in final_bar:
            images = images.to(device)
            flat = images.reshape(images.size(0), -1)

            optimizer.zero_grad()
            outputs = final_model(flat)
            loss = criterion(outputs, flat)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            final_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(final_loader)
        print(f"[Final Model] Epoch {epoch}/{num_epochs} | Loss: {epoch_loss:.4f}")

    print("\nModelo final treinado em todo o dataset de treino. ✅")
    return final_model


def vae_loss_function(recon_x, x, mu, logvar):
    """Compute the VAE loss."""

    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    batch_size = x.size(0)
    return (recon_loss + kld_loss)/batch_size


def train_variational_autoenconder_model_kfold(
    dataset,
    k=5,
    num_epochs=50,
    batch_size=64,
    lr=1e-3,
    latent_dim = 2,
    device="cpu"
):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    all_train_losses = []
    all_val_losses = []
    all_ssim = []

    print("========== TREINO K-FOLD VAE ==========")
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n=========== FOLD {fold+1}/{k} ===========")

        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Modelo do fold
        model = VAE(latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []
        ssim_list = []

        for epoch in range(1, num_epochs + 1):

            # --------------------------- TREINO ---------------------------
            model.train()
            running_loss = 0
            train_bar = tqdm(train_loader, desc=f"🔵 Treinando Fold {fold+1} Epoch {epoch}", leave=False)

            for images, _ in train_bar:
                images = images.to(device)

                optimizer.zero_grad()
                recon, mu, logvar = model(images)
                loss = vae_loss_function(recon, images, mu, logvar)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            # --------------------------- VALIDAÇÃO ---------------------------
            model.eval()
            running_val_loss = 0
            running_val_ssim = 0

            val_bar = tqdm(val_loader, desc=f"🟣 Validando Fold {fold+1} Epoch {epoch}", leave=False)
            with torch.no_grad():
                for images, _ in val_bar:
                    images = images.to(device)

                    # forward correto do VAE
                    recon, mu, logvar = model(images)

                    # mesma loss do treino
                    val_loss = vae_loss_function(recon, images, mu, logvar).item()
                    running_val_loss += val_loss

                    # SSIM
                    batch_ssim = compute_ssim_batch(recon, images)
                    running_val_ssim += batch_ssim

                    val_bar.set_postfix(
                        val_loss=f"{val_loss:.4f}",
                        ssim=f"{batch_ssim:.4f}"
                    )

            mean_val_loss = running_val_loss / len(val_loader)
            mean_ssim = running_val_ssim / len(val_loader)

            val_losses.append(mean_val_loss)
            ssim_list.append(mean_ssim)

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_ssim.append(ssim_list)

        torch.save(model.state_dict(), f"./vae_model_fold_{fold+1}.pth")

    print("\nK-Fold concluído!")
    save_metrics_plots(all_train_losses, all_val_losses, all_ssim, modelType= "vae")

    # ======================================================
    # TREINO FINAL EM TODO O DATASET
    # ======================================================
    print("\n========== TREINANDO MODELO FINAL ==========")

    final_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    final_model = VAE(latent_dim=2).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        final_model.train()
        running_loss = 0

        final_bar = tqdm(final_loader, desc=f"🔵 Treinando Modelo Final Epoch {epoch}", leave=False)
        for images, _ in final_bar:
            images = images.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = final_model(images)
            loss = vae_loss_function(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            final_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(final_loader)
        print(f"[Final Model] Epoch {epoch}/{num_epochs} | Loss: {epoch_loss:.4f}")

    print("\nModelo final treinado em todo o dataset. ✅")
    return final_model