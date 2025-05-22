import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# Utilities for point cloud processing (FPS, grouping)
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

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
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape.append(-1)
    idx = idx.reshape(B, -1)  # ← fixed here
    res = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    return res.view(*view_shape)


def query_ball_point(radius, nsample, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    group_idx = dist.argsort()[:, :, :nsample]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
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

    def forward(self, xyz, points):
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)  # (B, D, nsample, npoint)
        for conv in self.mlp_convs:
            new_points = F.relu(conv(new_points))
        new_points = torch.max(new_points, 2)[0]  # (B, D', npoint)
        new_xyz = new_xyz
        return new_xyz, new_points.permute(0, 2, 1)

class PointNetPlusPlus(nn.Module):
    def __init__(self, input_dim=3):  # input_dim = 3 for XYZ only, 6 for XYZ+RGB, etc.
        super().__init__()
        self.input_dim = input_dim - 3  # Only the additional features beyond XYZ

        self.sa1 = PointNetSetAbstraction(512, 0.2, 64, self.input_dim, [256, 256, 512])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 512, [512, 512, 1024])
        self.sa3 = PointNetSetAbstraction(1, None, None, 1024, [1024, 2048, 4096])

        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
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
        x = l3_points.squeeze(1)
        out = self.fc(x)
        return out
    
torch.backends.cuda.matmul.allow_tf32 = True  # Optional for performance
torch.backends.cudnn.benchmark = True  # Optional for performance

torch.cuda.set_per_process_memory_fraction(1.0, 0)
torch.cuda.empty_cache()

# Set max_split_size_mb to a smaller value to avoid fragmentation
# torch._C._cuda_set_allocator_config("max_split_size_mb:128")



model = PointNetPlusPlus(input_dim=6).cuda()
model.eval()

print("Num parameters:", sum(p.numel() for p in model.parameters()))
# exit()

# Warm up
# for _ in range(10):
#     pc = torch.randn(64, 8192, 6).cuda()
#     model(pc)

# Pre-allocate data
# pc = torch.randn(64, 8192, 6).cuda()
pc = torch.randn(1, 2**18, 6).cuda()  # 2^19 = 524288
times = []

for _ in range(10):
    _ = model(pc)
for _ in range(100):
    torch.cuda.synchronize()
    start = time.time()
    output = model(pc)
    torch.cuda.synchronize()
    times.append(time.time() - start)

print("Average inference time:", sum(times) / len(times))
plt.plot(times)
plt.xlabel('Iteration')
plt.ylabel('Time (s)')
plt.title('Inference Time per Iteration')
plt.show()