import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import eigh
import networkx as nx

# ========== Config ==========
SKELETON_DIR = "./nturgbd_skeletons"
NUM_JOINTS = 25
SEQUENCE_LENGTH = 100
NUM_CLASSES = 60
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== GFT Setup ==========
EDGES = [
    (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6),
    (6, 7), (20, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13),
    (13, 14), (0, 16), (16, 17), (17, 18), (0, 22), (22, 23), (23, 24)
]
G = nx.Graph()
G.add_edges_from(EDGES)
L = nx.laplacian_matrix(G, nodelist=range(NUM_JOINTS)).todense()
eigvals_np, eigvecs_np = eigh(L)
eigvecs = torch.tensor(eigvecs_np, dtype=torch.float32).to(DEVICE)  # (J, J)
eigvecs_inv = eigvecs.t()  # since eigenvectors are orthonormal

# ========== Skeleton Dataset ==========
def parse_skeleton_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_frames = int(lines[0])
    idx = 1
    frames = []
    for _ in range(num_frames):
        num_bodies = int(lines[idx])
        idx += 1
        if num_bodies == 0:
            continue
        idx += 6
        num_joints = int(lines[idx])
        idx += 1
        joints = []
        for _ in range(num_joints):
            joint_data = list(map(float, lines[idx].split()))
            joints.append(joint_data[:3])
            idx += 1
        frames.append(joints)
    return np.array(frames)  # (T, 25, 3)

def extract_class_from_filename(filename):
    base = os.path.basename(filename)
    class_id = int(base[1:4]) - 1  # A001 -> class 0
    return class_id

class NTUTransformerDataset(Dataset):
    def __init__(self, root, sequence_length=100):
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".skeleton")]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        joints = parse_skeleton_file(self.files[idx])  # (T, 25, 3)
        label = extract_class_from_filename(self.files[idx])
        if len(joints) < self.sequence_length:
            pad = self.sequence_length - len(joints)
            joints = np.pad(joints, ((0, pad), (0, 0), (0, 0)), mode='constant')
        else:
            joints = joints[:self.sequence_length]
        joints = joints[:, :, 0].T  # X-coordinates only (25, T)
        noisy = joints + np.random.normal(scale=0.3, size=joints.shape)
        return torch.tensor(noisy, dtype=torch.float32), torch.tensor(joints, dtype=torch.float32), label

# ========== Transformer Model with Integrated GFT ==========
class SkeletonTransformer(nn.Module):
    def __init__(self, joints=25, frames=100, d_model=128, nhead=8, num_layers=4, num_classes=60):
        super().__init__()
        self.gft = eigvecs  # (J, J)
        self.igft = eigvecs_inv  # (J, J)
        self.input_proj = nn.Linear(frames, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.reconstruction_head = nn.Linear(d_model, frames)
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):  # x: (B, J, T)
        # Apply GFT inside model
        x = torch.bmm(self.gft.unsqueeze(0).expand(x.size(0), -1, -1), x)  # (B, J, T)
        x = self.input_proj(x)  # (B, J, d_model)
        x = x.permute(1, 0, 2)  # (J, B, d_model)
        encoded = self.transformer(x)  # (J, B, d_model)
        encoded = encoded.permute(1, 0, 2)  # (B, J, d_model)
        recon = self.reconstruction_head(encoded)  # (B, J, T)
        # Apply inverse GFT
        recon = torch.bmm(self.igft.unsqueeze(0).expand(recon.size(0), -1, -1), recon)  # (B, J, T)
        class_logits = self.classification_head(encoded.permute(0, 2, 1))  # (B, num_classes)
        return recon, class_logits

# ========== Training Loop ==========
def train(model, dataloader, optimizer, criterion_recon, criterion_cls):
    model.train()
    total_loss = 0
    for noisy, clean, label in dataloader:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        recon, logits = model(noisy)
        loss_recon = criterion_recon(recon, clean)
        loss_cls = criterion_cls(logits, label)
        loss = loss_recon + loss_cls
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ========== Main ==========
if __name__ == '__main__':
    dataset = NTUTransformerDataset(SKELETON_DIR, sequence_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SkeletonTransformer(joints=NUM_JOINTS, frames=SEQUENCE_LENGTH, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_recon = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    for epoch in range(10):
        loss = train(model, dataloader, optimizer, criterion_recon, criterion_cls)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")

