import os
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torchvision.models as models


# Dataset
class KneeMRIDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.study_ids = df['StudyInstanceUID'].values

        self.labelsCols = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
                           'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
                           'Synovitis', "Baker's", 'Contusion', 'Fracture']

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = str(self.study_ids[idx])
        study_path = os.path.join(self.data_dir, study_id)

        slices = []
        try:
            for root, dirs, files in os.walk(study_path):
                for file in files:
                    if file.endswith(".dcm"):
                        slices.append(os.path.join(root, file))
            slices.sort()

            if len(slices) >= 3:
                idx1, idx2, idx3 = len(slices) // 4, len(slices) // 2, 3 * len(slices) // 4

                img1 = pydicom.dcmread(slices[idx1]).pixel_array.astype(np.float32)
                img2 = pydicom.dcmread(slices[idx2]).pixel_array.astype(np.float32)
                img3 = pydicom.dcmread(slices[idx3]).pixel_array.astype(np.float32)

                img1 = img1 / img1.max() if img1.max() > 0 else img1
                img2 = img2 / img2.max() if img2.max() > 0 else img2
                img3 = img3 / img3.max() if img3.max() > 0 else img3

                image = np.stack([img1, img2, img3], axis=0)
            else:
                image = np.zeros((3, 512, 512), dtype=np.float32)

        except Exception as e:
            image = np.zeros((3, 512, 512), dtype=np.float32)

        imageTensor = torch.tensor(image)

        if self.transform:
            imageTensor = self.transform(imageTensor)

        if all(col in self.df.columns for col in self.labelsCols):
            labels = self.df.iloc[idx][self.labelsCols].values.astype(np.float32)
            labelsTensor = torch.tensor(labels)
        else:
            labelsTensor = torch.zeros(12, dtype=torch.float32)

        return imageTensor, labelsTensor


class KneeModel(nn.Module):
    def __init__(self):
        super(KneeModel, self).__init__()

        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 12)

    def forward(self, x):
        return self.backbone(x)


# Training loop
if __name__ == "__main__":
    # Train data
    TRAIN_CSV_PATH = '/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv'
    TRAIN_DIR_PATH = '/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series.csv'

    labelsCols = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
                  'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
                  'Synovitis', "Baker's", 'Contusion', 'Fracture']

    df = pd.read_csv(TRAIN_CSV_PATH)

    df = df.dropna(subset=labelsCols)

    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = KneeMRIDataset(df_train, TRAIN_DIR_PATH)
    val_dataset = KneeMRIDataset(df_val, TRAIN_DIR_PATH)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    model = KneeModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_auc = 0.0
    model_save_path = "best_knee_model.pth"

    epochs = 30

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch ending | Avg Loss: {epoch_loss:.4f}")
        print("Starting validation")

        model.eval()
        all_preds = []
        all_true = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)

                outputs = torch.sigmoid(outputs)
                predsNumpy = outputs.cpu().numpy()
                labels_numpy = labels.cpu().numpy()

                all_preds.append(predsNumpy)
                all_true.append(labels_numpy)

        all_preds = np.vstack(all_preds)
        all_true = np.vstack(all_true)

        auc = 0.0
        try:
            auc = roc_auc_score(all_true, all_preds, average='macro')
            print(f"After epoch {epoch + 1} score | AUC: {auc:.4f}")
        except ValueError:
            print("Warning: Validation doesn't contain both classes for each abnormality")

        if auc > best_val_auc:
            torch.save(model.state_dict(), model_save_path)
            print(f"Best AUC yet, saving model to {model_save_path}")
            best_val_auc = auc
        else:
            print(f"AUC was better before, not saving model to {model_save_path}")

    print("Finished Training")