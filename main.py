import torch
import torch.nn as nn
from itertools import product
import argparse
from utils.dataloader import *
from utils.models import *
from tqdm import tqdm
from utils.utils import *
from copy import deepcopy
import math

HYPERPARAM_GRID = {
    "sgd": {
        "lr": [0.1, 0.05, 0.01],
        "momentum": [0.9],
        "weight_decay": [5e-4, 1e-4]
    },
    "adam": {
        "lr": [1e-3, 5e-4],
        "weight_decay": [1e-4, 0.0]
    }
}

MODEL_REGISTRY = {
    "resnet": ScalableResNet,
    "cnn": ScalableCNN,
    "mlp": ScalableMLP,
}

    
def train_network(
    model,
    model_kwargs: dict,
    train_loader,
    val_loader,
    device,
    optimizer_name: str,
    hyperparams: dict,
    max_epochs: int = 1000,
    criterion=nn.CrossEntropyLoss(),
    save_path: str = None
):
    
    """
    Trains a model using a small hyperparameter grid search until
    validation error reaches zero (interpolation point).

    Args:
        model_class: ScalableResNet / ScalableCNN / ScalableMLP
        model_kwargs: arguments to instantiate the model
        train_loader, val_loader: CIFAR-10 loaders
        device: torch.device
        max_epochs: upper bound on training epochs
        optimizer_choices: subset of ["sgd", "adam"]
        save_dir: where to save interpolated model

    Returns:
        dict with interpolation results
    """
    # -------------------------------------------------
    # Checkpoint exists → skip training
    # -------------------------------------------------
    if save_path is not None and is_final_checkpoint(save_path, device):
        print(f"[SKIP] Final checkpoint exists: {save_path}")
        return {
            "status": "skipped",
            "checkpoint": save_path
        }

    model = model_class(**model_kwargs).to(device)
    optimizer = build_optimizer(model, optimizer_name=optimizer_name, hyperparams=hyperparams)
    
    best_val_error = math.inf
    best_epoch = -1
    best_metrics = None
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            train_acc = correct / total
            pbar.set_postfix(
                train_loss=f"{loss_sum / len(train_loader):.4f}",
                train_acc=f"{train_acc:.4f}"
            )

        train_loss = loss_sum / len(train_loader)
        train_error = 1.0 - train_acc

        # --------------------
        # Validation
        # --------------------
        val_metrics = model.evaluate(val_loader, device, criterion)
        val_acc = val_metrics["accuracy"]
        val_error = val_metrics["error"]

        print(
            f"Epoch {epoch:04d} | "
            f"Train acc {train_acc:.4f}, error {train_error:.4f} | "
            f"Val acc {val_acc:.4f}, error {val_error:.4f}"
        )

        # --------------------
        # Best model tracking
        # --------------------
        if val_error < best_val_error:
            best_val_error = val_error
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_metrics = {
                "epoch": best_epoch,
                "train_acc": train_acc,
                "train_error": train_error,
                "val_acc": val_acc,
                "val_error": val_error,
                "num_parameters": model.num_parameters()
            }
            if save_path is not None:
                print(f"Saving checkpoint at {save_path}")
                model.save(save_path)

        # --------------------
        # Optional early stop at interpolation
        # --------------------
        if val_error == 0.0:
            print("✔ Interpolation achieved (zero validation error)")
            break
    # --------------------
    # Restore best model and save it with is_final flag
    # --------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path is not None:
            model.save(save_path, is_final=True)

    return {
        "status": "finished",
        "best_epoch": best_epoch,
        "best_val_error": best_val_error,
        "best_metrics": best_metrics,
        "checkpoint": save_path
    }


if __name__ == "__main__":
    args = parse_args()

    set_global_seed(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    model_class = MODEL_REGISTRY[args.model]

    # for cap in args.capacity:
    print("=" * 80)
    

    # ------------------------------
    # Data loaders
    # ------------------------------
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        seed=args.seed
    )
        
    # ------------------------------
    # Load or search hyperparameters
    # ------------------------------
    if args.mode == "search":
        print(f"HP SEARCH: model={args.model}, dataset={args.dataset}")

        small_train, small_val, _ = get_data_loaders(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            train_subset=0.2,
            seed=args.seed
        )
            
        best_cfg = run_hyperparam_search(
            model_class = model_class,
            capacities = args.capacity,
            train_loader = small_train,
            val_loader = small_val,
            hp_epochs = args.hp_epochs,
            device = device
        )

        save_fixed_hyperparams(
            args.dataset,
            args.model,
            best_cfg,
            args.capacity
        )
        
    elif args.mode == "train":
        print(f"TRAINING: model={args.model}, dataset={args.dataset}")
        
        fixed_cfg = load_fixed_hyperparams(args.dataset, args.model)
        assert fixed_cfg is not None, "Run hyperparameter search first."

        for i, cap in enumerate(args.capacity):
            ckpt_path = f"checkpoints_lr/{args.dataset}_{model_class.__name__}_{cap}.pt"

            if checkpoint_exists(ckpt_path):
                print(f"[SKIP] Checkpoint exists for capacity={cap}")
                continue
            
            model = model_class(capacity=args.capacity[i], num_classes=10)
            print(f"Training model: {cap} with {model.num_parameters()} params")
            
            result = train_network(
                model = model,
                model_kwargs={"capacity": cap, "num_classes": 10},
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                optimizer_name=fixed_cfg["optimizer"],
                hyperparams=fixed_cfg["hyperparams"],
                max_epochs=1000,
                save_path=ckpt_path
            )

            train_result_payload = {
                "capacity": cap,
                "num_parameters": result["best_metrics"]["num_parameters"],
                "best_epoch": result["best_epoch"],
                "train_acc": result["best_metrics"]["train_acc"],
                "train_error": result["best_metrics"]["train_error"],
                "val_acc": result["best_metrics"]["val_acc"],
                "val_error": result["best_metrics"]["val_error"],
                "checkpoint": result["checkpoint"]
            }
            

            save_experiment_result(
                dataset=args.dataset,
                model_name=args.model,
                capacity=cap,
                result_dict=train_result_payload,
                results_path="train_results_lr0.1.json"
            )

    elif args.mode == "test":
        print(f"TEST EVAL: model={args.model}, dataset={args.dataset} from saved checkpoints")
        # print("\nRunning test evaluation from saved checkpoints")

        for i, cap in enumerate(args.capacity):
            ckpt_path = f"checkpoints/{args.dataset}_{model_class.__name__}_{i}.pt"

            if not checkpoint_exists(ckpt_path):
                print(f"[SKIP] No checkpoint for capacity={cap}")
                continue

            model = model_class(capacity=cap, num_classes=10).to(device)
            model.load(ckpt_path, cap, map_location=device)

            test_metrics = model.evaluate(test_loader, device)

            test_result_payload = {
                "model ": i,
                "capacity": cap,
                "num_parameters": model.num_parameters(),
                "test": test_metrics
            }

            save_experiment_result(
                dataset=args.dataset,
                model_name=args.model,
                capacity=cap,
                result_dict=test_result_payload,
                results_path="test_results.json"
            )


        # ------------------------------
        # Train to interpolation
        # ------------------------------
        # result = train_to_interpolation(
        #     model_class=model_class,
        #     model_kwargs={
        #         "capacity": cap,
        #         "num_classes": 10
        #     },
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     device=device,
        #     optimizer_name=best_cfg["optimizer"],
        #     hyperparams=best_cfg["hyperparams"],
        #     max_epochs=1000,
        #     save_path=f"checkpoints/{args.model}_{args.dataset}_cap{cap}.pt"
        # )

        # print("Interpolation reached at epoch:", result["epoch"])
        # print("Final val error:", result["val_error"])

        # ------------------------------
        # Final test evaluation
        # ------------------------------
        # test_metrics = evaluate(
        #     result["model"],
        #     test_loader,
        #     device
        # )
        # test_metrics = model_class.evaluate(
        #     result["model"],
        #     test_loader,
        #     device
        # )

        # print("Test accuracy:", test_metrics["accuracy"])

