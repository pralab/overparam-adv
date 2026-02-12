import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
from autoattack import AutoAttack
from secml.ml.classifiers import CClassifierPyTorch
from secml.array import CArray
from secml.data import CDataset
from secml.ml.peval.metrics import CMetricAccuracy

from utils.dataloader import *
from utils.models import *
from utils.utils import *
from utils.attacks import *


EPS_REGISTRY = {
        "pgdl2":{
            ("mnist", "mlp"):   np.linspace(0.1, 3.0, 5),
            ("mnist", "cnn"):   np.linspace(0.01, 1.5, 5),
            ("cifar10", "resnet"): np.linspace(0.1, 6.0, 5)
        },
        "pgdlinf": {
            ("mnist", "mlp"):   np.linspace(0.01, 0.15, 5),
            ("mnist", "cnn"):   np.linspace(0.01, 0.07, 5),
            ("cifar10", "resnet"): np.linspace(0.1, 6.0, 5)
        },
        "autoattack": {
            ("mnist", "mlp"):   np.linspace(0.1, 2.0, 5),
            ("mnist", "cnn"):   np.linspace(0.01, 1.0, 5)
        }
}

MODEL_REGISTRY = {
    "resnet": ScalableResNet,
    "cnn": ScalableCNN,
    "mlp": ScalableMLP,
}
    


@torch.no_grad()
def eval_clean(model, x, y):
    model.eval()
    preds = model(x).argmax(dim=1)
    acc = (preds == y).float().mean().item()
    return {
        "accuracy": acc,
        "error_rate": 1.0 - acc
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_global_seed(args.seed)
    os.makedirs("results", exist_ok=True)
    output_path = f"results/sec_eval_{args.dataset}_{args.model}_{args.attack}_results.json"

    if args.dataset == "mnist":
        inp_size, in_channels, num_classes = 28, 1, 10
        capacities = range(1, 11)
        # capacities = [1, 4]
    elif args.dataset == "cifar10":
        inp_size, in_channels, num_classes = 32, 3, 10
        capacities = [1, 2, 4, 6, 8, 10, 16, 20, 24, 28]


    train_loader, _, test_loader = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        seed=args.seed,
        model_name=args.model,
        ds_normalization=False
    )

    x_test, y_test = collect_full_test_set(test_loader, device)

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "attack": args.attack,
        "results": {}
    }


    for i, cap in enumerate(capacities):
        print(f"[INFO] Evaluating cap={cap}")

        model_cls = MODEL_REGISTRY[args.model]
        model = model_cls(capacity=cap, num_classes=num_classes,
                          input_size=inp_size, in_channels=in_channels).to(device)
        print(f"\t {model.num_parameters()} params")
        ckpt_path = f"checkpoints/{args.dataset}_{model_cls.__name__}_{cap}.pt"
        print(f"\t Loading checkpoint from {ckpt_path}")
        model.load(ckpt_path, cap=cap, map_location=device)
        model.eval()

        cap_key = f"cap_{cap}"
        results["results"][cap_key] = {}

        # ---------------- CLEAN ----------------
        if args.attack == "clean":
            stats = eval_clean(model, x_test, y_test)
            results["results"][cap_key]["clean"] = stats
            continue

        # ---------------- ADVERSARIAL ----------------
        epsilons = EPS_REGISTRY[args.attack][(args.dataset, args.model)]
        
        if args.attack == "autoattack":
                
            stats = eval_autoattack_l2(
                args,
                model=model,
                test_loader=test_loader,
                epsilons=epsilons,
                device=device,
                batch_size=args.batch_size
            )
            for eps, acc in zip(epsilons, stats.values()):
                results["results"][cap_key][f"eps_{eps}"] = acc

        elif args.attack == "pgdl2":
            norm="l2"
            metric = CMetricAccuracy()
            print(f"[INFO] Running PGD-L2 for epsilons = {epsilons}")
            solver_params, _, lb, ub, y_target = get_pgd_attack_hyperparams(args.dataset)
            # lower, upper = get_normalized_bounds(args)
            stats = eval_secml_pgd(
                args,
                model=model,
                num_classes=num_classes,
                eps_list=epsilons,         # pass full list
                lower=lb,
                upper=ub,
                norm=norm,
                y_target=y_target,
                train_loader=train_loader,
                test_loader=test_loader,
                solver_params=solver_params,
                save_adv_ds=False
            )

            # stats.save_data(f"results/sec_eval_{args.dataset}_{args.model}_{cap{cap}}.gz")
            res = stats.sec_eval_data
            epsilons = res.param_values
            y_true = res.Y
            att_pred = res.Y_pred

            for i, eps in enumerate(epsilons):
                results["results"][cap_key][f"eps_{eps}"] = metric.performance_score(y_true=y_true, y_pred=att_pred[i])

        
        elif args.attack == "pgdlinf":
            print(f"[INFO] Running Foolbox PGD-L∞ for epsilons = {epsilons}")

            stats = eval_foolbox_pgd_linf(
                model=model,
                test_loader=test_loader,
                epsilons=epsilons,
                device=device,
                steps=100 if args.dataset == "mnist" else 150
            )

            for eps, acc in stats.items():
                results["results"][cap_key][f"eps_{eps}"] = acc

            
            
        # ---------------- Save ----------------
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[✓] Results saved to {output_path}")


if __name__ == "__main__":
    main()