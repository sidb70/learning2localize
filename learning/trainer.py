'''
Trainer class for training 3D NNs for apple localization. 

This class is responsible for training the model, validating it, and saving the best model.

'''


import os
import uuid
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datatools import ApplePointCloudDataset, pad_collate_fn
from tqdm import tqdm
from models import (
    PointNetPlusPlus,
)
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
DATA_ROOT = os.path.join(PROJECT_ROOT, "blender/dataset/raw/apple_orchard-5-20-processed")
TRAIN_MAN = os.path.join(PROJECT_ROOT, "blender/dataset/curated/apple-orchard-v1/train.jsonl")
TEST_MAN  = os.path.join(PROJECT_ROOT, "blender/dataset/curated/apple-orchard-v1/test.jsonl")


class Trainer:
    """Trainer class for training 3D NNs for apple localisation with LR scheduling."""

    def __init__(self, model, train_dataset, val_dataset, config, num_workers: int = 12):
        # --- configuration / bookkeeping ----------------------------------------------------
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.batch_size = config['batch_size']
        self.grad_accum_steps = config['grad_accum_steps']

        # --- model / optimiser --------------------------------------------------------------
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=config.get("learning_rate", 1e-3),
                                    weight_decay=config.get("weight_decay", 1e-4))

        # --- scheduler ----------------------------------------------------------------------
        self.scheduler = self._build_scheduler(config)

        # --- loss ---------------------------------------------------------------------------
        beta = config.get("smooth_l1_beta", 0.005)
        self.criterion = nn.SmoothL1Loss(beta=beta)

        # --- data ---------------------------------------------------------------------------
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_collate_fn,  # handles variable point cloud sizes
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=pad_collate_fn,  # handles variable point cloud sizes
        )

        # --- misc ---------------------------------------------------------------------------
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:8]
        self.run_dir = os.path.join(config.get("log_dir", "./logs"), run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # --- tensorboard writer ---
        self.writer = SummaryWriter(log_dir=self.run_dir, filename_suffix="_apple_localization")
        self.best_val_loss = float("inf")

    # -------------------------------------------------------------------------------------
    # Scheduler factory
    # -------------------------------------------------------------------------------------
    def _build_scheduler(self, cfg):
        sched_type = cfg.get("lr_scheduler", "plateau").lower()
        if sched_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.get("step_size", 2),
                gamma=cfg.get("lr_gamma", 0.6),
            )
        if sched_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.get("t_max", cfg.get("num_epochs", 100)),
                eta_min=cfg.get("min_lr", 1e-6),
            )
        # default: ReduceLROnPlateau
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.get("lr_factor", 0.5),
            patience=cfg.get("lr_patience", 3),
            min_lr=cfg.get("min_lr", 1e-6),
        )

    # -------------------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------------------
    def train(self):
        num_epochs = self.cfg.get("num_epochs", 10)
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_z_err = 0.0
            accum_steps = 0

            for i, (clouds, centers, mask,aux) in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            ):
                # move to device
                clouds = clouds.to(self.device)
                centers = centers.to(self.device)
                mask = mask.to(self.device)
                norm_scale = aux["norm_scale"].to(self.device).view(-1, 1)

                # build labels (only z)
                labels = centers[:, 2].unsqueeze(1).float()

                # forward / backward --------------------------------------------------
                outputs = self.model(clouds, mask)
                loss = self.criterion(outputs, labels) 
                if self.grad_accum_steps>0:
                    loss = loss / self.grad_accum_steps
                loss.backward()

                # real‑space error for monitoring
                z_err_m = (outputs - labels).abs() * norm_scale
                epoch_z_err += z_err_m.mean().item()

                accum_steps += 1
                epoch_loss += loss.item()
                if i %10 == 0:
                    print(f"Batch {i} | Loss: {loss.item():.4f} | ZErr: {z_err_m.mean().item():.4f} m")

                if accum_steps >= self.grad_accum_steps:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accum_steps = 0

            # catch leftover grads ----------------------------------------------------
            if accum_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # average metrics ---------------------------------------------------------
            epoch_loss /= len(self.train_loader)
            epoch_z_err /= len(self.train_loader)

            # validation --------------------------------------------------------------
            val_loss, val_z_err = self.validate()

            # scheduler step ----------------------------------------------------------
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # logging ----------------------------------------------------------------
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Loss/train", epoch_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("ZErr/train", epoch_z_err, epoch)
            self.writer.add_scalar("ZErr/val", val_z_err, epoch)
            self.writer.add_scalar("LR", lr, epoch)

            print(
                f"Epoch {epoch+1:03d} | LR {lr:.2e} | TrainLoss {epoch_loss:.4f} | ValLoss {val_loss:.4f} | ValZerr {val_z_err:.4f} m"
            )

            # save best ---------------------------------------------------------------
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.cfg.get("log_dir", "./logs"), "best_model.pth"))

        self.writer.close()
        print("Training complete. Best val loss = {:.4f}".format(self.best_val_loss))

    # -------------------------------------------------------------------------------------
    # Validation loop
    # -------------------------------------------------------------------------------------
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        val_z_err = 0.0
        with torch.no_grad():
            for clouds, centers, mask, aux in self.val_loader:
                clouds = clouds.to(self.device)
                centers = centers.to(self.device)
                mask = mask.to(self.device)
                norm_scale = aux["norm_scale"].to(self.device).view(-1, 1)

                labels = centers[:, 2].unsqueeze(1).float()
                outputs = self.model(clouds, mask)

                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                z_err_m = (outputs - labels).abs() * norm_scale
                val_z_err += z_err_m.mean().item()

        val_loss /= len(self.val_loader)
        val_z_err /= len(self.val_loader)
        return val_loss, val_z_err
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")
    def test(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                clouds, centers, aux = batch
                clouds = clouds.to(self.device)
                centers = centers.to(self.device)
                occ = aux['occ_rate'].unsqueeze(1).float().to(self.device)

                # concat centers and occ_batch to a labels tensor of shape (B, 4)
                # labels = torch.cat([centers, occ], dim=1)
                z_vals = centers[:,2].unsqueeze(1).float()
                labels = z_vals

                outputs = self.model(clouds)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f"Test loss: {test_loss:.4f}")
    def log_metrics(self, metrics):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.epoch)
        print(f"Metrics logged: {metrics}")
    def close(self):
        self.writer.close()
        print("TensorBoard writer closed.")
    def __del__(self):
        self.close()
        print("Trainer object deleted.")

if __name__ == "__main__":


    torch.cuda.empty_cache()

    # Example usage
    config = {
        "learning_rate": 1e-2,
        "lr_scheduler": "step",   # step | cosine | plateau
        "lr_factor": 0.5,
        "lr_patience": 3,
        "min_lr": 1e-6,
        'weight_decay': 1e-4,
        'batch_size': 16,
        'voxel_size': 0.0025,
        'num_epochs': 40,
        'grad_accum_steps': 1,  # set to >1 for gradient accumulation
        'num_workers': 12,
        'log_dir': './logs',
    }

    DATA_ROOT = os.path.join(PROJECT_ROOT, "blender/dataset/")


    model = PointNetPlusPlus(input_dim=6, output_dim=1).cuda()
    # init wieghts
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    train_dataset = ApplePointCloudDataset(data_root=DATA_ROOT, 
                                           manifest_path=TRAIN_MAN,
                                           config=config,
                                           augment=True)
    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    test_dataset = ApplePointCloudDataset(data_root=DATA_ROOT, 
                                          manifest_path=TEST_MAN,
                                            config=config,
                                            augment=False)

    trainer = Trainer(model, train_dataset, val_dataset, config)
    trainer.train()
    trainer.test(test_dataset)
    trainer.save_model(os.path.join(config['log_dir'], 'best_model.pth'))

    trainer.close()
