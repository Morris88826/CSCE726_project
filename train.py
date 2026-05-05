import os
import time
import yaml
import json
import torch
import logging
import random
import argparse
import numpy as np
from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from libs.model import LinearClassifier
from libs.sampler import Sampler
from libs.logger import setup_logging
from libs.trainer import train_one_epoch, evaluate
from libs.dataloader import build_dataloaders
from libauc.losses import EntLossClassification
from libauc.optimizers import SCENT
from torch.nn import CrossEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Set random seed to {seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with SCENT loss')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--model', type=str, default='LinearClassifier', help='Model name (default: LinearClassifier)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    sampling_strategy = config.get("sampler", None)
    if sampling_strategy is None:
        raise ValueError("Please specify a sampling strategy in the config file under the 'sampler' key. Options: 'all', 'batch', 'curriculum', 'global', 'local'.")


    name = args.model + "_" + os.path.basename(args.config).replace(".yaml", "")
    root_out_dir = config["root_out_dir"]
    out_dir = Path(root_out_dir) / name
    out_log_file = out_dir / "out.log"
    out_ckpt_dir = out_dir / "checkpoints"
    os.makedirs(out_ckpt_dir, exist_ok=True)

    # save config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)


    setup_logging(out_log_file)
    set_seed(config["seed"])
    
    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    train_loader, val_loader, test_loader = build_dataloaders(data_dir, batch_size)

    model = LinearClassifier(feature_dim=config["feature_dim"], num_classes=config["num_classes"])
    model = model.to(device)

    sampler = Sampler(
        prototypes_path="./data/prototypes.pkl",
        num_classes=config["num_classes"],
        n_clusters=config.get("subgroup_size", 1000),
        num_negatives=config.get("num_negatives", 128),
        seed=config["seed"],
    )

    if config["optimizer"] == "SGD":
        optimizer = SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    elif config["optimizer"] == "scent" or config["optimizer"] == "sox":
        optimizer = SCENT(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    

    epochs = config["epochs"]
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    is_scent = config["optimizer"] == "scent"
    if is_scent:
        criterion = EntLossClassification(data_size=config['data_size'], alpha=config['alpha'], is_scent=is_scent, alpha_multiplier=config['alpha_multiplier'])
    else:
        criterion = EntLossClassification(data_size=config['data_size'], gamma=config['gamma'], is_scent=False)

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_test_loss = float("inf")
    start_epoch = 0

    same_budget = config.get("same_budget", True)
    for epoch in range(start_epoch + 1, epochs + 1):
        logging.info(f"Epoch {epoch}/{epochs}, learning_rate={lr_scheduler.get_last_lr()[0]:.6f}")
        if hasattr(criterion, "adjust_gamma"):
            criterion.adjust_gamma(epoch, epochs)
        cross_entropy_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion, sampler, sampling_strategy=sampling_strategy, device=device, same_budget=same_budget)
        lr_scheduler.step()

        logging.info("Evaluating on validation set")
        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            logging.info("New best model found")
            best_val_acc = val_acc
            if test_loader is not None:
                logging.info("Evaluating on test set")
                best_test_loss, best_test_acc = evaluate(model, test_loader, device)
        logging.info(f"Epoch {epoch}: cross_entropy_loss={cross_entropy_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.6f}")
        logging.info(f"  Best val_acc={best_val_acc:.6f}")

        if test_loader is not None:
            logging.info(f"  Best test acc={best_test_acc:.6f} Best test loss={best_test_loss:.6f}")
        eval_results = {
            "epoch": epoch,
            "cross_entropy_loss": cross_entropy_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
        }
        if test_loader is not None:
            eval_results.update({
                "best_test_acc": best_test_acc,
                "best_test_loss": best_test_loss,
            })
        with open(out_dir / f"eval_{name}.jsonl", "a") as f:
            f.write(json.dumps(eval_results) + "\n")

        if epoch % config['save_frequency'] == 0 or epoch == epochs:
            save_dict = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
            }
            if hasattr(criterion, "nu"):
                save_dict["criterion_nu"] = criterion.nu.cpu()
            torch.save(save_dict, out_dir / "checkpoints" / f"epoch_{epoch}.pt")
            logging.info(f"Saved checkpoint for epoch {epoch}.")