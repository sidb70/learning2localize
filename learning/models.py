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
    idx = idx.reshape(B, -1) 
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
class PointNetSetAbstractionMSG(nn.Module):
    def __init__(self, npoint, radii, nsamples, in_channel, mlp_channels_list):
        """
        Parameters
        ----------
        npoint : int
            Number of centroids to sample (FPS).
        radii : list of float
            List of radii for ball query at each scale.
        nsamples : list of int
            Number of neighbors to query per scale.
        in_channel : int
            Input feature channels (excluding XYZ).
        mlp_channels_list : list of list of int
            One MLP configuration per scale, e.g., [[32, 32, 64], [64, 64, 128]].
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlp_channels_list)

        self.npoint = npoint
        self.radii = radii
        self.nsamples = nsamples
        self.branches = nn.ModuleList()

        for i in range(len(radii)):
            mlp = mlp_channels_list[i]
            convs = nn.Sequential()
            last_channel = in_channel + 3  # +3 for local xyz coords
            for j, out_channel in enumerate(mlp):
                convs.add_module(f"conv_{j}", nn.Conv2d(last_channel, out_channel, 1))
                convs.add_module(f"relu_{j}", nn.ReLU())
                last_channel = out_channel
            self.branches.append(convs)

    def forward(self, xyz, points, mask):
        """
        xyz    : (B, N, 3)
        points : (B, N, D) or None
        mask   : (B, N) bool
        Returns:
        new_xyz    : (B, S, 3)
        new_points : (B, S, sum_out_channels)
        """
        B, N, _ = xyz.shape
        device = xyz.device
        if mask is None:
            # fallback to old behaviour
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # ---- sampling --------------------------------------------------
        fps_idx = farthest_point_sample_masked(xyz, mask, self.npoint)  # (B, S)
        new_xyz = index_points(xyz, fps_idx)                             # (B, S, 3)

        # ---- multi‑scale grouping --------------------------------------
        grouped_outputs = []
        for radius, nsample, mlp_conv in zip(self.radii, self.nsamples, self.branches):
            group_idx = query_ball_point(radius, nsample, xyz, new_xyz, mask)  # (B, S, nsample)
            grouped_xyz = index_points(xyz, group_idx)                         # (B, S, nsample, 3)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points, group_idx)               # (B, S, nsample, D)
                grouped_features = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                grouped_features = grouped_xyz_norm                           # (B, S, nsample, 3)

            # (B, S, nsample, C) → (B, C, nsample, S)
            grouped_features = grouped_features.permute(0, 3, 2, 1)
            out = mlp_conv(grouped_features)                                  # (B, D_out, nsample, S)
            out = torch.max(out, dim=2)[0]                                    # (B, D_out, S)
            grouped_outputs.append(out)

        # Concatenate features from all scales
        new_points = torch.cat(grouped_outputs, dim=1)                         # (B, sum_D_out, S)
        return new_xyz, new_points.permute(0, 2, 1)   
class PointNetPlusPlus(nn.Module):
    def __init__(self, input_dim=3, output_dim=3,npoints = [1024,256,64], radii=[0.005, 0.1, 0.3], nsamples=[64, 128, 256],
                    mlp_channels=[[128, 128, 256], [256, 256, 512], [512, 512, 1024]]):
            
        super().__init__()
        self.input_dim = input_dim - 3  # Only the additional features beyond XYZ

        self.sa1 = PointNetSetAbstraction(npoint=npoints[0], radius=radii[0], nsample=nsamples[0], in_channel=self.input_dim, mlp=mlp_channels[0])
        self.sa2 = PointNetSetAbstraction(npoint=npoints[1], radius=radii[1], nsample=nsamples[1], in_channel=mlp_channels[0][-1], mlp=mlp_channels[1])
        self.sa3 = PointNetSetAbstraction(npoint=npoints[2], radius=radii[2], nsample=nsamples[2], in_channel=mlp_channels[1][-1], mlp=mlp_channels[2])

        self.npoints = npoints  # Store npoints for mask calculation
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        xyz = x[:, :, :3]
        rgb = x[:, :, 3:] if x.size(2) > 3 else None
        
        if mask is None:
            B, N, _ = xyz.shape
            mask = torch.ones(B, N, dtype=torch.bool, device=xyz.device)
        
        l1_xyz, l1_rgb = self.sa1(xyz, rgb, mask)
        
        # Calculate proper downsampling ratio for mask1
        original_points = mask.size(1)  # Should be your input point count
        l1_points = self.npoints[0]  # 512 in your new config
        downsample_ratio1 = original_points // l1_points
        mask1 = F.max_pool1d(mask.unsqueeze(1).float(), downsample_ratio1).squeeze(1).bool()
        
        l2_xyz, l2_rgb = self.sa2(l1_xyz, l1_rgb, mask1)
        
        # Calculate proper downsampling ratio for mask2
        downsample_ratio2 = l1_points // self.npoints[1]  # 512 // 128 = 4
        mask2 = F.max_pool1d(mask1.unsqueeze(1).float(), downsample_ratio2).squeeze(1).bool()
        
        l3_xyz, l3_rgb = self.sa3(l2_xyz, l2_rgb, mask2)
        
        max_pool = l3_rgb.max(dim=1)[0]
        global_features = max_pool
        return self.fc(global_features)
class PointNetPlusPlusMSG(PointNetPlusPlus):
    def __init__(self, input_dim=3, output_dim=3,
                 npoints=[1024, 256, 64],
                 radii=[[0.005, 0.1, 0.2], [0.01, 0.2, 0.3], [0.02, 0.3, 0.5]],
                 nsamples = [[32, 64, 128], [32, 64, 128], [32, 64, 128]],
                    mlp_channels_list=[[ [32, 32, 64], [64, 64, 128], [64, 96, 128] ],
                                        [ [64, 64, 128], [128, 128, 256], [128, 128, 256] ],
                                        [ [128, 196, 256], [196, 196, 256], [196, 256, 512] ]]):
        super().__init__(input_dim, output_dim)
        def _total_out_channels(branches):
            return sum(mlp[-1] for mlp in branches)

        # inside __init__ of PointNetPlusPlusMSG
        sa1_out_dim = _total_out_channels(mlp_channels_list[0])
        sa2_out_dim = _total_out_channels(mlp_channels_list[1])
        sa3_out_dim = _total_out_channels(mlp_channels_list[2])

        self.sa1 = PointNetSetAbstractionMSG(
            npoint=npoints[0],
            radii=radii[0],
            nsamples=nsamples[0],
            in_channel=self.input_dim,
            mlp_channels_list=mlp_channels_list[0]
        )

        self.sa2 = PointNetSetAbstractionMSG(
            npoint=npoints[1],
            radii=radii[1],
            nsamples=nsamples[1],
            in_channel=sa1_out_dim,
            mlp_channels_list=mlp_channels_list[1]
        )

        self.sa3 = PointNetSetAbstractionMSG(
            npoint=npoints[2],
            radii=radii[2],
            nsamples=nsamples[2],
            in_channel=sa2_out_dim,
            mlp_channels_list=mlp_channels_list[2]
        )


def farthest_point_sample_unmasked(xyz, npoint):
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



def query_ball_point_unmasked(radius, nsample, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    group_idx = dist.argsort()[:, :, :nsample]
    return group_idx

def sample_and_group_unmasked(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample_unmasked(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point_unmasked(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


class PointNetSetAbstractionUnmasked(nn.Module):
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

    def forward(self, xyz, points):
        new_xyz, new_points = sample_and_group_unmasked(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)  # (B, D, nsample, npoint)
        new_points = new_points.float()
        for conv in self.mlp_convs:
            new_points = F.relu(conv(new_points))
        new_points = torch.max(new_points, 2)[0]  # (B, D', npoint)
        new_xyz = new_xyz
        return new_xyz, new_points.permute(0, 2, 1)

class PointNetPlusPlusUnmasked(nn.Module):
    def __init__(self, input_dim=3, output_dim=3,npoints = [1024,256,64], radii=[0.02, 0.05, 0.12], nsamples=[64, 64, 128],
                    mlp_channels=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]]):
            
        super().__init__()
        self.input_dim = input_dim - 3  # Only the additional features beyond XYZ

        self.sa1 = PointNetSetAbstractionUnmasked(npoint=npoints[0], radius=radii[0], nsample=nsamples[0], in_channel=self.input_dim, mlp=mlp_channels[0])
        self.sa2 = PointNetSetAbstractionUnmasked(npoint=npoints[1], radius=radii[1], nsample=nsamples[1], in_channel=mlp_channels[0][-1], mlp=mlp_channels[1])
        self.sa3 = PointNetSetAbstractionUnmasked(npoint=npoints[2], radius=radii[2], nsample=nsamples[2], in_channel=mlp_channels[1][-1], mlp=mlp_channels[2])

        self.fc = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, N, input_dim)
        First 3 dims are XYZ. Remaining are features (optional).
        """
        xyz = x[:, :, :3]
        points = x[:, :, 3:] if x.shape[2] > 3 else None

        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        max_pool = l3_points.max(dim=1)[0]
        mean_pool = l3_points.mean(dim=1)
        global_features = torch.cat([max_pool, mean_pool], dim=1)  # (B, 2048)
        return self.fc(global_features)  # (B, output_dim)

        
if __name__ == "__main__":

    # ------------- runtime hints -----------------------------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------- network -----------------------------------------------
    model = PointNetPlusPlusUnmasked(input_dim=6, output_dim=2,
                        npoints=[512, 128, 32],
                        radii=[0.01, 0.05, 0.15],
                        nsamples=[32, 64, 128],
                        mlp_channels=[
                            [64, 64, 128],
                            [128, 128, 256], 
                            [256, 256, 512]  
                        ]).cuda().eval()

    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # ------------- warm‑up ----------------------------------------------
    pcs = [torch.randn(1, torch.randint(1000, 4000, ()), 6, device=device) for _ in range(10)]
    for i in range(10):
        _ = model(pcs[i])
    print("Warmup done.")

    # ------------- benchmark --------------------------------------------
    pc = torch.randn(1, 4096, 6, device=device)  # (B, N, D)
    times = []
    for i in range(100):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model(pc)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    print(f"Average inference time: {sum(times)/len(times):.6f} s")

    plt.plot(times)
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.title("Inference time per iteration")
    plt.savefig('/home/keyi/sid/learning2localize/learning/pointnetpp_inference_time.png')
    plt.close()