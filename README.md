# Autoencoder & VAE — Fashion-MNIST

Implementação de um **Autoencoder (AE)** e de um **Variational Autoencoder (VAE)**
em **PyTorch** para reconstrução e geração de imagens do dataset **Fashion-MNIST**.
O treinamento usa **K-Fold Cross-Validation** e a avaliação calcula métricas de
reconstrução como **MSE** e **SSIM**. O projeto é totalmente operável por linha
de comando (subcomandos `train` e `generate`) e empacotável via **Docker**.

Trabalho da disciplina de **Deep Learning (2025.3)** do **PESC — Programa de
Engenharia de Sistemas e Computação / COPPE-UFRJ**.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)
![uv](https://img.shields.io/badge/deps-uv-DE5FE9)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

---

## Funcionalidades

- **Dois modelos:** Autoencoder clássico (`ae`) e Variational Autoencoder (`vae`), selecionáveis por flag.
- **K-Fold Cross-Validation** no conjunto de treino, com treino do modelo final e avaliação no conjunto de teste (split 80/20, `random_state=42`).
- **Métricas de reconstrução:** MSE e SSIM (estrutura preparada para incluir MAE, RMSE e PSNR).
- **Geração de imagens:** subcomando `generate` que carrega um modelo salvo e produz reconstruções/amostras em `outputs/`.
- **Uso de GPU automático:** usa CUDA quando disponível, caso contrário CPU.
- **Reprodutibilidade:** dependências travadas com `uv` (`uv.lock`) e ambiente containerizado via `Dockerfile` + `makefile`.
- **Download automático** do Fashion-MNIST via `torchvision` (salvo em `data/`).

---

## 📁 Estrutura do projeto

```
Deep-Learning-2025.3/
├── autoenconders/            # pacote principal
│   ├── cli.py                # interface de linha de comando (argparse)
│   ├── model.py              # definição das redes: Autoencoder e VAE
│   ├── train.py              # laços de treino (AE e VAE com K-Fold)
│   ├── test.py               # avaliação nos dados de teste
│   └── evaluate.py           # geração/visualização de resultados
├── data/FashionMNIST/raw/    # dataset (baixado automaticamente)
├── metrics/                  # métricas e gráficos gerados
├── modelTrain/               # artefatos de treinamento
├── outputs/                  # imagens reconstruídas/geradas
├── main.py                   # ponto de entrada -> autoenconders.cli:main
├── model.pth                 # pesos de um modelo treinado
├── pyproject.toml            # metadados e dependências
├── uv.lock                   # lockfile do uv
├── Dockerfile                # imagem de execução
└── makefile                  # atalhos de build/run com Docker
```

---

## ⚙️ Instalação

Requer **Python 3.13**. Recomenda-se o [`uv`](https://docs.astral.sh/uv/) para
gerenciar o ambiente (o repositório já traz `pyproject.toml` e `uv.lock`):

```bash
# clonar
git clone https://github.com/ThallesNonato1123/Deep-Learning-2025.3.git
cd Deep-Learning-2025.3

# criar o ambiente e instalar as dependências travadas
uv sync
```

<details>
<summary>Alternativa sem <code>uv</code> (pip)</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.6.0" "torchvision>=0.24.1" "scikit-image>=0.25.2" \
            "scikit-learn>=1.7.2" "matplotlib>=3.10.7" "tqdm>=4.67.1"
```
</details>

---

## ▶️ Uso

O ponto de entrada é o `main.py`, que expõe dois subcomandos: `train` e `generate`.
Com `uv`, basta prefixar os comandos com `uv run`.

### Treinar

```bash
# Autoencoder clássico
python main.py train --model-type ae --k 5 --epochs 50 --batch-size 64 --lr 0.001

# Variational Autoencoder
python main.py train --model-type vae --latent-dim 2 --epochs 50
```

| Argumento       | Tipo  | Default     | Descrição                                        |
|-----------------|-------|-------------|--------------------------------------------------|
| `--model-type`  | str   | `ae`        | Modelo a treinar: `ae` ou `vae`                  |
| `--k`           | int   | `5`         | Número de folds do K-Fold Cross-Validation       |
| `--epochs`      | int   | `50`        | Número de épocas de treinamento                  |
| `--batch-size`  | int   | `64`        | Tamanho do batch do DataLoader                   |
| `--lr`          | float | `0.001`     | Learning rate do otimizador Adam                 |
| `--latent-dim`  | int   | `2`         | Dimensão do espaço latente (usado no VAE)        |
| `--save-path`   | str   | `model.pth` | Caminho para salvar os pesos do modelo treinado  |

### Gerar imagens a partir de um modelo salvo

```bash
python main.py generate --model-path model.pth --model-type ae --num-images 8
```

| Argumento       | Tipo  | Default      | Descrição                                   |
|-----------------|-------|--------------|---------------------------------------------|
| `--model-path`  | str   | *(obrigat.)* | Caminho do modelo salvo (`.pth`)            |
| `--model-type`  | str   | *(obrigat.)* | `ae` ou `vae`                               |
| `--latent-dim`  | int   | `2`          | Dimensão latente (deve casar com o treino)  |
| `--num-images`  | int   | `8`          | Quantidade de imagens a gerar               |
| `--output-dir`  | str   | `outputs`    | Pasta de saída das imagens                  |
| `--filename`    | str   | `output.png` | Nome do arquivo de saída                    |

---

## 🐳 Docker

O `makefile` encapsula os comandos de container:

```bash
make build   # constrói a imagem (autoenconder:latest)
make run     # constrói e executa o container
make shell   # abre um shell dentro do container
make clean   # remove a imagem
```

---

## 📊 Métricas

A avaliação da reconstrução usa:

- **MSE** (Mean Squared Error) — erro quadrático médio pixel a pixel.
- **SSIM** (Structural Similarity Index) — similaridade estrutural (via `scikit-image`), mais alinhada à percepção humana que o MSE.

A estrutura permite estender facilmente para **MAE**, **RMSE** e **PSNR**.
Gráficos e artefatos de avaliação são salvos em `metrics/` e as imagens
reconstruídas/geradas em `outputs/`.

---

## 🗂️ Dataset

**Fashion-MNIST** — 70.000 imagens em tons de cinza (28×28) de 10 categorias de
vestuário. É baixado automaticamente pelo `torchvision` na primeira execução e
armazenado em `data/`.

---

## 👤 Autoria

**Thalles Nonato** — Deep Learning (2025.3), PESC/COPPE-UFRJ

- GitHub: [@ThallesNonato1123](https://github.com/ThallesNonato1123)

> 💡 O pacote está nomeado como `autoenconders` (com a grafia usada no código).
> Manter esse nome preserva os imports; renomear exigiria ajustar `main.py`
> e o `cli.py`.
