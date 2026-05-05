import torch
import logging
import torch.nn.functional as F

def train_one_epoch(epoch, model, loader, optimizer, criterion, sampler, sampling_strategy, device, same_budget=True):
    model.train()
    total_loss = 0.0
    total = 0
    for i, batch in enumerate(loader):
        feats, labels, indices = batch
        feats = feats.to(device)
        if sampling_strategy == "all":
            logits = model(feats, labels, classes="all")

        elif sampling_strategy == "batch":
            logits = model(feats, labels, classes=None)
        else:
            sampled_classes = sampler.sample_classes(
                strategy=sampling_strategy,
                batch_classes=labels,
                device=device,
                epoch=epoch,
                same_budget=same_budget,
            )
            logits = model(feats, labels, classes=sampled_classes)
        if sampling_strategy == "all":
            loss = F.cross_entropy(logits, labels.to(device, dtype=torch.long))
            loss_dict = {"loss": loss}
        else:
            loss_dict = criterion(logits, indices)
        loss = loss_dict["loss"]
        with torch.no_grad():
            model.eval()
            labels = labels.to(device, dtype=torch.long)
            cross_entropy_loss = F.cross_entropy(model.fc(feats), labels)
            loss_dict["cross_entropy_loss"] = cross_entropy_loss
            model.train()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += cross_entropy_loss.item() * feats.size(0)
        total += feats.size(0)

        if i % 100 == 0:
            log_str = f"  Batch {i} / {len(loader)}:"
            for key, value in loss_dict.items():
                log_str += f" {key}={value.item():.6f}"
            logging.info(log_str)

    return total_loss / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for i, batch in enumerate(loader):
        feats, labels, _ = batch
        feats = feats.to(device)
        labels = labels.to(device, dtype=torch.long)

        logits = model.fc(feats)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * feats.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += feats.size(0)

        if i % 100 == 0:
            logging.info(f"  Batch {i} / {len(loader)}: loss={loss.item():.6f}")

    return total_loss / total, correct / total