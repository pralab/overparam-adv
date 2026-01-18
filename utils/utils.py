import json
import os
from utils.models import *
import argparse
import torch.optim as optim
import torch.nn.functional as F

MODEL_REGISTRY = {
    "resnet": ScalableResNet,
    "cnn": ScalableCNN,
    "mlp": ScalableMLP,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpolation and overparameterization experiments"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["search", "train", "test"],
        required=True,
        help="Run hyperparameter search or perform training"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_REGISTRY.keys(),
        required=True,
        help="Model architecture"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "mnist"],
        required=True,
        help="Dataset name"
    )

    parser.add_argument(
        "--capacity",
        type=int,
        nargs="+",
        required=True,
        help="Model capacity (single int or list for sweep)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--hp_epochs",
        type=int,
        default=30,
        help="Number of epochs for hyperparameter search"
    )

    parser.add_argument(
        "--train_subset",
        type=float,
        default=1.0,
        help="Fraction of training data to use"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    return parser.parse_args()

def checkpoint_exists(path):
    return os.path.isfile(path)

def build_optimizer(model, optimizer_name, hyperparams):
    if optimizer_name.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=hyperparams["lr"],
            momentum=hyperparams.get("momentum", 0.0),
            weight_decay=hyperparams.get("weight_decay", 0.0),
            nesterov=hyperparams.get("nesterov", False)
        )

    elif optimizer_name.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams.get("weight_decay", 0.0)
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def search_run(
    model,
    train_loader,
    val_loader,
    device,
    optimizer_name,
    hyperparams,
    epochs
):
    model.to(device)

    optimizer = build_optimizer(model, optimizer_name, hyperparams)

    best_val_acc = 0.0

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        val_metrics = model.evaluate(val_loader, device)
        best_val_acc = max(best_val_acc, val_metrics["accuracy"])

    return {"best_val_accuracy": best_val_acc}

def is_final_checkpoint(path: str, device) -> bool:
    if not os.path.exists(path):
        return False
    try:
        ckpt = torch.load(path, map_location=device)
        return ckpt.get("is_final", False) is True
    except Exception:
        return False

def _make_model_dataset_key(dataset, model_name):
    return f"{dataset}::{model_name}"

def load_fixed_hyperparams(dataset, model_name, registry_path="best_hyperparams.json"):
    if not os.path.exists(registry_path):
        return None

    with open(registry_path, "r") as f:
        registry = json.load(f)

    return registry.get(_make_model_dataset_key(dataset, model_name), None)

def save_fixed_hyperparams(
    dataset,
    model_name,
    best_config,
    capacities,
    registry_path="best_hyperparams.json"
):
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    registry[_make_model_dataset_key(dataset, model_name)] = {
        **best_config,
        "searched_capacities": capacities
    }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

def run_hyperparam_search(
    model_class,
    capacities,
    train_loader,
    val_loader,
    device,
    hp_epochs
):
    candidates = [
        ("sgd", {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4}),
        ("sgd", {"lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4}),
        ("adam", {"lr": 1e-3, "weight_decay": 1e-4}),
    ]

    best_val_acc = -float("inf")
    best_cfg = None

    for opt_name, hparams in candidates:
        print(f"Testing hyperparams: {opt_name} | {hparams}")

        for cap in capacities:
            print(f"  Capacity={cap}")

            model = model_class(
                capacity=cap,
                num_classes=10
            ).to(device)

            optimizer = build_optimizer(model, opt_name, hparams)

            local_best_acc = 0.0

            for epoch in range(hp_epochs):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = torch.nn.functional.cross_entropy(model(x), y)
                    loss.backward()
                    optimizer.step()

                val_metrics = model.evaluate(val_loader, device)
                local_best_acc = max(local_best_acc, val_metrics["accuracy"])

            print(f"    best val acc={local_best_acc:.4f}")

            if local_best_acc > best_val_acc:
                best_val_acc = local_best_acc
                best_cfg = {
                    "optimizer": opt_name,
                    "hyperparams": hparams
                }

    print("✓ Selected hyperparameters based on best val accuracy")
    print(best_cfg)
    return best_cfg

def _make_result_key(dataset, model_name, capacity):
    return f"{dataset}::{model_name}::cap{capacity}"


def save_experiment_result(
    dataset,
    model_name,
    capacity,
    result_dict,
    results_path="results.json"
):
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    key = _make_result_key(dataset, model_name, capacity)
    registry[key] = result_dict

    with open(results_path, "w") as f:
        json.dump(registry, f, indent=2)


