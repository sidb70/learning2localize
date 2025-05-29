import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(xyz.device)
    distance = torch.ones(B, N).to(xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long).to(xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        dist = dist.to(distance.dtype)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def square_distance(src, dst):
    # src : (B, N, 3)   dst : (B, M, 3)
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist                                          # (B, N, M)

# --------------------------- FPS with mask -------------------------------
def farthest_point_sample_masked(xyz, mask, npoint):
    """
    xyz   : (B, N, 3)    float32
    mask  : (B, N)       bool, 1 = valid
    Return: (B, npoint)  long indices in the *original* N‑length dimension.
    """
    B, N, _ = xyz.shape
    device  = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)

    # init distances: 0 for valid, +inf for padded → they will never be chosen
    distance = torch.full((B, N), 1e10, device=device)
    distance[mask] = 0.0

    # pick the first farthest randomly *among valid points*
    farthest = torch.multinomial(mask.float(), 1).squeeze(-1)  # (B,)
    batch_idx = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # only update those distances where the point is valid
        mask_update = (dist < distance) & mask
        distance[mask_update] = dist[mask_update]
        farthest = torch.max(distance, -1)[1]

    return centroids                                          # (B, npoint)


def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape.append(-1)
    idx = idx.reshape(B, -1)  # ← fixed here
    res = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    return res.view(*view_shape)


# --------------------------- radius query with mask ----------------------
def query_ball_point(radius, nsample, xyz, new_xyz, mask=None):
    """
    Returns (B, S, nsample) indices into xyz.
    If a neighborhood has <nsample valid points, it is padded
    by repeating the first valid index (PointNet++ original trick).
    """
    dist = square_distance(new_xyz, xyz)                      # (B, S, N)
    # set distance of invalid points to +inf so they are never chosen
    if mask is not None:
        inv_mask = (~mask).unsqueeze(1).expand_as(dist)
        dist = dist.masked_fill(inv_mask, 1e10)

    idx = dist.argsort()[:, :, :nsample]                      # nearest first
    if mask is not None:
        first = idx[:, :, 0].unsqueeze(-1).expand(-1, -1, nsample)
        # if there were fewer than nsample valid neighbours, repeat first
        valid_counts = mask.sum(dim=1, keepdim=True)              # (B,1)
        overflowing = valid_counts < nsample
        if overflowing.any():
            idx = torch.where(overflowing.unsqueeze(-1), first, idx)
    return idx                                                # (B, S, nsample)

def sample_and_group(npoint, radius, nsample, xyz, points, mask=None):
    """
    mask: (B, N) bool — 1=real point, 0=padding.
    """
    if mask is None:
        # fallback to old behaviour
        B, N, _ = xyz.shape
        mask = torch.ones(B, N, dtype=torch.bool, device=xyz.device)

    # ---- sampling --------------------------------------------------------
    fps_idx  = farthest_point_sample_masked(xyz, mask, npoint)      # (B,S)
    new_xyz  = index_points(xyz, fps_idx)                           # (B,S,3)

    # ---- grouping --------------------------------------------------------
    group_idx   = query_ball_point(radius, nsample, xyz, new_xyz, mask)
    grouped_xyz = index_points(xyz, group_idx)                      # (B,S,nsample,3)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)           # local coords

    if points is not None:
        grouped_points = index_points(points, group_idx)            # (B,S,nsample,D')
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm                               # (B,S,nsample,3)

    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            last_channel = out_channel

    def forward(self, xyz, points, mask):
        new_xyz, new_points = sample_and_group(
            self.npoint, self.radius, self.nsample,
            xyz, points, mask
        )      
        new_points = new_points.permute(0, 3, 2, 1)  # (B, D, nsample, npoint)
        new_points = new_points.float()
        for conv in self.mlp_convs:
            new_points = F.relu(conv(new_points))
        new_points = torch.max(new_points, 2)[0]  # (B, D', npoint)
        new_xyz = new_xyz
        return new_xyz, new_points.permute(0, 2, 1)

class PointNetPlusPlus(nn.Module):
    def __init__(self, input_dim=3, output_dim=3,npoints = [1024,256,64], radii=[0.02, 0.05, 0.12], nsamples=[64, 64, 128],
                    mlp_channels=[[128, 128, 256], [256, 256, 512], [512, 512, 1024]]):
            
        super().__init__()
        self.input_dim = input_dim - 3  # Only the additional features beyond XYZ

        self.sa1 = PointNetSetAbstraction(npoint=npoints[0], radius=radii[0], nsample=nsamples[0], in_channel=self.input_dim, mlp=mlp_channels[0])
        self.sa2 = PointNetSetAbstraction(npoint=npoints[1], radius=radii[1], nsample=nsamples[1], in_channel=mlp_channels[0][-1], mlp=mlp_channels[1])
        self.sa3 = PointNetSetAbstraction(npoint=npoints[2], radius=radii[2], nsample=nsamples[2], in_channel=mlp_channels[1][-1], mlp=mlp_channels[2])

        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x, mask):
        xyz    = x[:, :, :3]
        points = x[:, :, 3:] if x.size(2) > 3 else None

        l1_xyz, l1_points = self.sa1(xyz, points, mask)
        # down‑sample mask to l1 (take OR over each group of 4)
        mask1 = F.max_pool1d(mask.unsqueeze(1).float(), 4).squeeze(1).bool()

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, mask1)
        mask2 = F.max_pool1d(mask1.unsqueeze(1).float(), 4).squeeze(1).bool()

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, mask2)
        max_pool = l3_points.max(dim=1)[0]
        # mean_pool = l3_points.mean(dim=1)
        # global_features = torch.cat([max_pool, mean_pool], dim=1)  # (B, 2048)
        global_features = max_pool  # Use only max pooling for simplicity
        return self.fc(global_features)  # (B, output_dim)
if __name__ == "__main__":

    # ------------- runtime hints -----------------------------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------- network -----------------------------------------------
    model = PointNetPlusPlus(
        input_dim=6, output_dim=3,
        npoints=[2048, 512, 128],            # 4× reduction each stage
        radii=[0.02, 0.05, 0.12],
        nsamples=[64, 64, 128],
        mlp_channels=[[128, 128, 256],
                      [256, 256, 512],
                      [512, 512, 1024]],
    ).to(device).eval()

    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M")

    # ------------- warm‑up ----------------------------------------------
    B_warm, N_warm = 32, 8192          # <- satisfies N/4 == 2048
    pc_warm  = torch.randn(B_warm, N_warm, 6, device=device)
    mask_warm = torch.ones(B_warm, N_warm, dtype=torch.bool, device=device)
    for _ in range(10):
        _ = model(pc_warm, mask_warm)

    # ------------- benchmark --------------------------------------------
    B, N = 32, 8192
    pc   = torch.randn(B, N, 6, device=device)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)

    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model(pc, mask)                   # pass the mask every time
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    print(f"Average inference time: {sum(times)/len(times):.6f} s")

    plt.plot(times)
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.title("Inference time per iteration")
    plt.savefig('/home/keyi/sid/learning2localize/learning/pointnetpp_inference_time.png')
    plt.close()