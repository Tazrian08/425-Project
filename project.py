import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.manifold import TSNE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = load_dataset("ag_news", split="train[:2000]")
texts = dataset["text"]
true_labels = dataset["label"]


embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts, convert_to_tensor=True)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.embeddings[idx]


class Autoencoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


autoencoder = Autoencoder().to(device)
dataloader = DataLoader(EmbeddingDataset(embeddings), batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print("\nTraining Autoencoder...")
for epoch in range(20):
    autoencoder.train()
    total_loss = 0
    for x, _ in dataloader:
        x = x.to(device).float()
        x_hat = autoencoder(x)
        loss = loss_fn(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


autoencoder.eval()
with torch.no_grad():
    latent = autoencoder.encoder(embeddings.to(device).float()).cpu().numpy()


kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_preds = kmeans.fit_predict(latent)

sil_kmeans = silhouette_score(latent, kmeans_preds)
db_kmeans = davies_bouldin_score(latent, kmeans_preds)
ch_kmeans = calinski_harabasz_score(latent, kmeans_preds)
print(f"\nKMeans:")
print(f"  Silhouette Score: {sil_kmeans:.4f}")
print(f"  Davies-Bouldin Index: {db_kmeans:.4f}")
print(f"  Calinski-Harabasz Index: {ch_kmeans:.4f}")


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, embedding_dim):
        super().__init__()
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, embedding_dim))
        nn.init.xavier_uniform_(self.clusters.data)

    def forward(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.clusters)**2, dim=2))
        q = q ** ((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

class DEC(nn.Module):
    def __init__(self, autoencoder, n_clusters=4):
        super().__init__()
        self.encoder = autoencoder.encoder
        self.clustering_layer = ClusteringLayer(n_clusters, 64)

    def forward(self, x):
        z = self.encoder(x)
        q = self.clustering_layer(z)
        return q, z

def target_distribution(q):
    weight = q ** 2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()


dec = DEC(autoencoder, n_clusters=4).to(device)
dec.clustering_layer.clusters.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device)

optimizer = torch.optim.Adam(dec.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.KLDivLoss(reduction='batchmean')

# DEC training
print("\nTraining DEC...")
for epoch in range(20):
    dec.train()
    total_loss = 0
    for x, _ in dataloader:
        x = x.to(device).float()
        q, _ = dec(x)
        p = target_distribution(q).detach()
        loss = loss_fn(torch.log(q), p)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, KL Loss: {total_loss:.4f}")


dec.eval()
with torch.no_grad():
    final_latent = dec.encoder(embeddings.to(device).float()).cpu()
    q = dec.clustering_layer(final_latent)
    dec_preds = torch.argmax(q, dim=1).numpy()

sil_dec = silhouette_score(final_latent, dec_preds)
db_dec = davies_bouldin_score(final_latent, dec_preds)
ch_dec = calinski_harabasz_score(final_latent, dec_preds)
print(f"\nDEC:")
print(f"  Silhouette Score: {sil_dec:.4f}")
print(f"  Davies-Bouldin Index: {db_dec:.4f}")
print(f"  Calinski-Harabasz Index: {ch_dec:.4f}")


def visualize(latents, preds, title):
    reduced = PCA(n_components=2).fit_transform(latents)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=preds, cmap='tab10', s=10)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

visualize(latent, kmeans_preds, "KMeans Clustering on AG News Embeddings")
visualize(final_latent.numpy(), dec_preds, "DEC Clustering on AG News Embeddings")

def visualize_tsne(latents, preds, title):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=preds, cmap='tab10', s=10)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()


visualize_tsne(latent, kmeans_preds, "KMeans Clustering (t-SNE)")
visualize_tsne(final_latent.numpy(), dec_preds, "DEC Clustering (t-SNE)")


def plot_label_distribution(labels, title):
    plt.figure(figsize=(5,3))
    plt.hist(labels, bins=np.arange(5)-0.5, rwidth=0.8, color='skyblue', edgecolor='black')
    plt.xticks(range(4))
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title(title)
    plt.show()

plot_label_distribution(dataset["label"], "Train Set Label Distribution")


test_dataset = load_dataset("ag_news", split="test[:2000]")
plot_label_distribution(test_dataset["label"], "Test Set Label Distribution")

# pip install torch datasets sentence-transformers scikit-learn matplotlib