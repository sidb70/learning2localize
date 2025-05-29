from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray
import torch, numpy as np
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from datatools import ApplePointCloudDataset, pad_collate_fn
from models import PointNetPlusPlus
import dotenv 
import os 

# Load environment variables from .env file one directory up
dotenv.load_dotenv(dotenv.find_dotenv())
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
SEED = int(os.getenv("SEED", 42))  # default to 42 if not set
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)  # set seed for all GPUs



DATA_ROOT = os.path.join(PROJECT_ROOT, "blender/dataset/")
TRAIN_MAN = os.path.join(PROJECT_ROOT, "blender/dataset/curated/apple-orchard-v1/train.jsonl")
TEST_MAN  = os.path.join(PROJECT_ROOT, "blender/dataset/curated/apple-orchard-v1/test.jsonl")
# ------------------------------------------------------------------
def train_val(cfg: dict) -> dict:
    """
    Train PointNet++ on an Apple dataset with the hyper‑params in `cfg`
    and return the validation metric needed by the HPO engine.

    Returns
    -------
    dict  e.g. {"val_mm": 0.0063, "val_loss": 0.00012}
    """

    device = "cuda" 
    # ------------- datasets ---------------------------------------
    ds_full = ApplePointCloudDataset(
        data_root     = cfg["data_root"],
        manifest_path = cfg["train_manifest"],
        augment       = cfg.get("augment", True),
        config        = cfg,                      # pass voxel_size, etc.
    )


    dataset_size = cfg.get("dataset_size", .1)
    num_samples = int(len(ds_full) * dataset_size)
    indices = np.random.choice(len(ds_full), num_samples, replace=False)
    ds_full = torch.utils.data.Subset(ds_full, indices)



    val_len = int(len(ds_full) * cfg.get("val_split", 0.2))
    train_len = len(ds_full) - val_len
    ds_train, ds_val = random_split(ds_full, [train_len, val_len])

    dl_train = DataLoader(
        ds_train,
        batch_size   = cfg["batch_size"],
        shuffle      = True,
        num_workers  = cfg.get("num_workers", 8),
        pin_memory   = True,
        collate_fn   = pad_collate_fn,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size   = cfg["batch_size"],
        shuffle      = False,
        num_workers  = cfg.get("num_workers", 12),
        pin_memory   = True,
        collate_fn   = pad_collate_fn,
    )

    # ------------- model / optim ----------------------------------
    model = PointNetPlusPlus(input_dim=6, output_dim=1).to(device)
    for m in model.modules():                      # Xavier init
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    criterion = nn.SmoothL1Loss(beta=cfg.get("smooth_l1_beta", 0.005))
    optimizer  = optim.AdamW(
        model.parameters(),
        lr           = cfg["learning_rate"],
        weight_decay = cfg.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size   = cfg.get("step_size", 2),
        gamma       = cfg.get("gamma", 0.1),
    )

    # ------------- training loop ----------------------------------
    num_epochs = cfg.get("budget_epochs", cfg["num_epochs"])
    for epoch in range(num_epochs):
        model.train()
        for i, (clouds, centers, mask, aux) in enumerate(dl_train):
            clouds, centers, mask = clouds.to(device), centers.to(device), mask.to(device)
            labels = centers[:, 2:3]

            optimizer.zero_grad()
            outputs = model(clouds, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Batch {i+1}/{len(dl_train)}, "
                  f"Loss: {loss.item():.6f}"
                  , end="\r")
        print()

        # ----------- quick val every epoch ------------------------
        model.eval()
        val_loss, val_mm, count = 0.0, 0.0, 0
        with torch.no_grad():
            for i, (clouds, centers, mask, aux) in enumerate(dl_val):
                clouds, centers, mask = clouds.to(device), centers.to(device), mask.to(device)
                labels = centers[:, 2:3]
                outputs = model(clouds, mask)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)

                norm_scale = aux["norm_scale"].to(device).view(-1, 1)
                val_mm += ((outputs - labels).abs() * norm_scale * 1000).sum().item()  # mm
                count  += labels.size(0)

                print(f"Epoch {epoch+1}/{num_epochs}, Val Batch {i+1}/{len(dl_val)}, "
                      f"Loss: {loss.item():.6f}, MM: {val_mm/count:.6f}", end="\r")


        val_loss /= count
        val_mm   /= count
        scheduler.step(val_loss)

        # optional: early stop if already under target
        if val_mm < cfg.get("target_mm", 5.0):
            break

    return {"val_mm": val_mm, "val_loss": val_loss}



SEARCH_SPACE = {
    # ---------- data / pre‑proc -----------------------------------
    "voxel_size": tune.choice([0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.03, 0.05]),
    # ---------- optimiser ----------------------------------------
    "learning_rate": tune.loguniform(1e-3, 5e-2),
    "weight_decay":  tune.loguniform(1e-6, 1e-3),
    # ---------- scheduler ----------------------------------------
    "step_size": tune.choice([1, 2, 5]),
    "gamma":     tune.uniform(0.1, 0.9),
    # ---------- model / loss -------------------------------------
    "smooth_l1_beta": tune.uniform(0.001, 0.01),
    }
STATIC_ARGS = {
    "data_root":      DATA_ROOT,
    "train_manifest": TRAIN_MAN,
    "val_split":      0.2,          # 20 % of subset goes to validation
    "dataset_size":   0.15,         # use 10 % of full dataset for speed
    "batch_size":     16,           # batch size for training
    "num_workers":    12,
    "num_epochs":     40,           # upper bound for ASHA
    "budget_epochs":  15,           # will be truncated by ASHA’s max_t
    "augment":        True,
}
def train_tune(tune_cfg):
    """Merge static + sampled cfg, run training, report val_mm to Tune."""
    full_cfg = dict(STATIC_ARGS)           # copy static keys
    full_cfg.update(tune_cfg)              # add sampled keys

    metrics = train_val(full_cfg)          
    tune.report({"val_mm": metrics["val_mm"], "val_loss": metrics["val_loss"]})


# ------------------------------------------------------------------
# Example sweep call
if __name__ == "__main__":
    # cfg = {
    #     'dataset_size': 0.1,  # Use 10% of the dataset for quick testing
    #     # --- fixed paths ---
    #     "data_root":       DATA_ROOT,
    #     "train_manifest":  TRAIN_MAN,
    #     # --- search knobs ---
    #     "voxel_size":   0.003,
    #     "learning_rate":1e-3,
    #     "weight_decay": 1e-4,
    #     "step_size":   2,       
    #     "gamma": 0.1,
    #     "batch_size":   16,
    #     "num_epochs":   40,
    #     "budget_epochs":10,     # ← short budget for ASHA first rung
    # }
    # metrics = train_val(cfg)
    # print(metrics)

    ray.init(num_cpus=32, num_gpus=2)      

    # ASHA: promote top configs; kill bad ones quickly
    scheduler = ASHAScheduler(
        max_t=15,          # corresponds to full_cfg["num_epochs"]
        grace_period=4,    # allow at least 5 epochs before stopping
        reduction_factor=3,
        metric="val_mm",   # the metric to optimize
        mode="min"         # minimize val_mm (lower is better)
    )

    analysis = tune.run(
        train_tune,
        config=SEARCH_SPACE,
        num_samples=100,                  # total trials
        scheduler=scheduler,
        storage_path=os.path.join(PROJECT_ROOT,'learning/hpo_results'),
        name="apple_pointnet_hpo",
        resources_per_trial={"cpu": 32, "gpu": 2},   # each trial gets 1 GPU
    )

    print("Best config:", analysis.best_config)
    print("Best val_mm:", analysis.best_result["val_mm"])


