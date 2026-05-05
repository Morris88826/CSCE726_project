import os
import tqdm
import pickle
import numpy as np
from libs.dataloader import build_dataloaders
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
import torch

class Sampler:
    def __init__(
        self,
        prototypes_path,
        num_classes,
        n_clusters=1000,
        num_negatives=128,
        seed=0,
    ):
        if prototypes_path is None or not os.path.exists(prototypes_path):
            raise FileNotFoundError(f"Prototypes file {prototypes_path} not found.")
        with open(prototypes_path, 'rb') as f:
            data = pickle.load(f)

        self.prototypes = data['prototypes']
        valid_classes = data['valid_classes']
        print(f"Loaded prototypes from {prototypes_path}: {self.prototypes.shape}")

        self.valid_classes = np.asarray(valid_classes, dtype=np.int64)
        self.num_classes = num_classes
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=10000,
            max_iter=100,
            random_state=seed,
        )

        self.subgroup_ids = kmeans.fit_predict(self.prototypes)

        self.class_to_subgroup = {}
        for cls, gid in zip(self.valid_classes, self.subgroup_ids):
            self.class_to_subgroup[int(cls)] = int(gid)

        self.subgroup_to_classes = defaultdict(list)
        for cls, gid in self.class_to_subgroup.items():
            self.subgroup_to_classes[gid].append(cls)

        self.subgroup_to_classes = {
            gid: np.asarray(classes, dtype=np.int64)
            for gid, classes in self.subgroup_to_classes.items()
        }

    def lambda_schedule(self, epoch, t1=3, t2=15, lambda_max=0.7):
        if epoch <= t1:
            return 0.0
        elif epoch >= t2:
            return lambda_max
        else:
            return lambda_max * (epoch - t1) / (t2 - t1)

    def sample_classes(
        self,
        strategy,
        batch_classes,
        device,
        epoch=None,
        lambda_t=None,
        include_batch_classes=True,
        num_negatives=None,
        same_budget=False,
    ):
        """
        Returns sampled class IDs for model(..., classes=classes).

        strategy:
            "all"        -> use all classes
            "batch"      -> use only in-batch unique labels
            "global"     -> random negatives from full label space
            "local"      -> negatives from same subgroup
            "curriculum" -> mixture of global + local
        """

        if strategy == "all":
            return "all"

        if isinstance(batch_classes, torch.Tensor):
            batch_np = batch_classes.detach().cpu().numpy().astype(np.int64)
            batch_size = batch_classes.numel()
        else:
            batch_np = np.asarray(batch_classes, dtype=np.int64)
            batch_size = len(batch_np)

        batch_unique = np.unique(batch_np)
        batch_set = set(batch_unique.tolist())

        if strategy == "batch":
            return torch.as_tensor(batch_unique, dtype=torch.long, device=device)

        # decide number of negatives
        if same_budget:
            # total sampled classes ≈ batch size
            target_total_classes = batch_size
            num_neg = max(0, target_total_classes - len(batch_unique))
        else:
            # fixed number of negatives, e.g., 1024
            num_neg = self.num_negatives if num_negatives is None else num_negatives

        if strategy == "curriculum":
            if lambda_t is None:
                if epoch is None:
                    raise ValueError("For curriculum sampling, provide either epoch or lambda_t.")
                lambda_t = self.lambda_schedule(epoch)
        elif strategy == "global":
            lambda_t = 0.0
        elif strategy == "local":
            lambda_t = 1.0
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        sampled = set()

        max_trials = max(1, num_neg * 10)
        trials = 0

        while len(sampled) < num_neg and trials < max_trials:
            trials += 1

            # choose one positive label from current batch
            y = int(self.rng.choice(batch_unique))

            use_local = self.rng.random() < lambda_t

            if use_local and y in self.class_to_subgroup:
                gid = self.class_to_subgroup[y]
                candidates = self.subgroup_to_classes[gid]
                neg = int(self.rng.choice(candidates))
            else:
                neg = int(self.rng.choice(self.valid_classes))

            # avoid positives and duplicates
            if neg not in batch_set:
                sampled.add(neg)

        if len(sampled) > 0:
            sampled = np.asarray(list(sampled), dtype=np.int64)

            if include_batch_classes:
                sampled_classes = np.unique(np.concatenate([batch_unique, sampled]))
            else:
                sampled_classes = np.unique(sampled)
        else:
            sampled_classes = batch_unique

        return torch.as_tensor(sampled_classes, dtype=torch.long, device=device)
    
# python -m libs.sampler
if __name__ == "__main__":
    prototypes_path = './data/prototypes.pkl'
    if os.path.exists(prototypes_path):
        print(f"Prototypes file {prototypes_path} already exists. Skipping prototype generation.")
    else:
        data_dir = "./features/treeoflife10m_subset"
        batch_size = 128
        feature_dim = 512
        num_classes = 163002

        train_loader, val_loader, test_loader = build_dataloaders(data_dir, batch_size)
        sums = np.zeros((num_classes, feature_dim), dtype=np.float32)
        counts = np.zeros(num_classes, dtype=np.int64)
        for x, y, _ in tqdm.tqdm(train_loader):
            y = y.numpy()
            sums[y] += x.numpy()
            counts[y] += 1

        valid_classes = np.where(counts > 0)[0]
        print(f"Found {len(valid_classes)} valid classes with at least one sample.")
        prototypes = sums[valid_classes] / counts[valid_classes, None]
        prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
        print(prototypes.shape)
        out_path = './data/prototypes.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump({
                'prototypes': prototypes,
                'valid_classes': valid_classes,
            }, f)
        print(f"Saved prototypes to {out_path}")