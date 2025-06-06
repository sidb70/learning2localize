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
import json
from models import (
    PointNetPlusPlus,
    PointNetPlusPlusUnmasked
)
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
SEED = int(os.getenv("SEED", 42))
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
            # collate_fn=pad_collate_fn,  # handles variable point cloud sizes
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            # collate_fn=pad_collate_fn,  # handles variable point cloud sizes
        )

        # --- misc ---------------------------------------------------------------------------
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:8]
        self.run_dir = os.path.join(config.get("log_dir", "./logs"), run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # --- tensorboard writer ---
        self.writer = SummaryWriter(log_dir=self.run_dir, filename_suffix="_apple_localization")
        self.best_val_loss = float("inf")


        # save hyperparameters
        self.writer.add_text("hyperparameters", str(self.cfg), 0)
        # save config dict to json
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.cfg, f, indent=4)

    # -------------------------------------------------------------------------------------
    # Scheduler factory
    # -------------------------------------------------------------------------------------
    def _build_scheduler(self, cfg):
        sched_type = cfg.get("lr_scheduler", None)
        if sched_type is None:
            return None
        sched_type = sched_type.lower()
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

            for i, (clouds, centers,aux) in enumerate(
                tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            ):
                # move to device
                clouds = clouds.to(self.device)
                centers = centers.to(self.device)
                # mask = mask.to(self.device)
                norm_scale = aux["norm_scale"].to(self.device).view(-1, 1)
                occ_rate = aux["occ_rate"].to(self.device).view(-1, 1)

                # build labels (only z)
                labels = centers[:, 2].unsqueeze(1).float()
                # concat centers and occ_rate to a labels 
                labels = torch.cat([labels, occ_rate], dim=1)
                labels = labels.to(self.device)

                # forward / backward --------------------------------------------------
                # exit()
                outputs = self.model(clouds)#, mask)
                loss = self.criterion(outputs, labels) 
                if self.grad_accum_steps>0:
                    loss = loss / self.grad_accum_steps
                loss.backward()

                # real‑space error for monitoring
                pred_z = outputs[:, 0].unsqueeze(1)  # only z value
                label_z = labels[:, 0].unsqueeze(1)  # only z value
                z_err_m = (pred_z - label_z).abs() * norm_scale
                epoch_z_err += z_err_m.mean().item()

                epoch_loss += loss.item()
                if i %10 == 0:
                    print(f"Batch {i} | Loss: {loss.item():.4f} | ZErr: {z_err_m.mean().item():.4f} m")

                if accum_steps >= self.grad_accum_steps:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accum_steps = 0
                    print(f"Step {i} | Optimizer step taken. Grad accum steps: {self.grad_accum_steps}")
                else:
                    # accumulate gradients
                    # print(f"Step {i} | Gradients accumulated. Current accum steps: {accum_steps + 1}")
                    accum_steps += 1

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
            if self.scheduler is not None:
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
                torch.save(self.model.state_dict(), os.path.join(self.run_dir, "best_model.pth"))

        self.writer.close()
        print("Training complete. Best val loss = {:.4f}".format(self.best_val_loss))

    # -------------------------------------------------------------------------------------
    # Validation loop
    # -------------------------------------------------------------------------------------
    def validate(self):
        print("Validating...")
        self.model.eval()
        val_loss = 0.0
        val_z_err = 0.0
        with torch.no_grad():
            for clouds, centers, aux in self.val_loader:
                clouds = clouds.to(self.device)
                centers = centers.to(self.device)
                # mask = mask.to(self.device)
                norm_scale = aux["norm_scale"].to(self.device).view(-1, 1)
                occ_rate = aux["occ_rate"].to(self.device).view(-1, 1)

                labels = centers[:, 2].unsqueeze(1).float()
                labels  = torch.cat([labels, occ_rate], dim=1)
                labels = labels.to(self.device)
                outputs = self.model(clouds)#, mask)

                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # real‑space error for monitoring
                pred_z = outputs[:, 0].unsqueeze(1)
                label_z = labels[:, 0].unsqueeze(1)
                z_err_m = (pred_z - label_z).abs() * norm_scale
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
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=pad_collate_fn)
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                clouds, centers, masks, aux = batch
                clouds = clouds.to(self.device)
                masks = masks.to(self.device)
                centers = centers.to(self.device)
                occ_rate = aux['occ_rate'].unsqueeze(1).float().to(self.device)

                # concat centers and occ_batch to a labels tensor of shape (B, 4)
                labels = centers[:,2].unsqueeze(1).float()
                labels = torch.cat([labels, occ_rate], dim=1)

                outputs = self.model(clouds, masks)
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
    import argparse

    parser = argparse.ArgumentParser(description="Train a PointNet++ model for apple localization.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a checkpoint to load the model from.")
    args = parser.parse_args()

    ckpt = args.ckpt

    torch.cuda.empty_cache()
    # Example usage
    config ={
        "learning_rate": 0.0005,
        "lr_scheduler": "step",
        "step_size": 5,
        "lr_gamma": 0.65,
        "t_max": 100,
        "smooth_l1_beta": 0.005,
        "lr_factor": 0.5,
        "lr_patience": 10,
        "min_lr": 1e-05,
        "weight_decay": 0.0001,
        "batch_size": 1,
        "voxel_size": 0.0045,
        "num_epochs": 500,
        "grad_accum_steps": 64,
        "num_workers": 12,
        "log_dir": "./logs/fixed_dataset"
    }

    DATA_ROOT = os.path.join(PROJECT_ROOT, "blender/dataset/raw/apple_orchard-5-20-fp-only")
    TRAIN_MAN = os.path.join(PROJECT_ROOT, "blender/dataset/curated/apple-orchard-v3-fp-only-ignore-narrow-box/train.jsonl")
    TEST_MAN  = os.path.join(PROJECT_ROOT, "blender/dataset/curated/apple-orchard-v3-fp-only-ignore-narrow-box/test.jsonl")


    model = PointNetPlusPlusUnmasked(input_dim=6, output_dim=2,
                        npoints=[512, 128, 32],
                        radii=[0.01, 0.05, 0.15],
                        nsamples=[32, 64, 128],
                        mlp_channels=[
                            [64, 64, 128],
                            [128, 128, 256], 
                            [256, 256, 512]  
                        ]).cuda()
    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location='cuda'))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")
    # init weights
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
    trainer.load_model(trainer.run_dir + '/best_model.pth')
    print("Starting testing...")
    trainer.test(test_dataset)
    # trainer.save_model(os.path.join(trainer.run_dir, 'best_model.pth'))

    trainer.close()
