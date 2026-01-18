import torch
import torch.optim
from secml.ml import CClassifierPyTorch

from tinynet import ExpandableCNN_mnist, ExpandableFcReLu_mnist, RandomFourier, ResNet
from folder import CNN_MNIST, FC_MNIST, RF_CIFAR10, RESN_CIFAR10

def get_models_and_path(ds, network, expansions):
    if ds == "mnist":
        if network == "cnn":
            model_folder = CNN_MNIST
            clf_names = [f"clf_mnist-cnn_{i}.gz" for i in expansions]
            nn_module = ExpandableCNN_mnist
        elif network == "fcrelu":
            model_folder = FC_MNIST
            clf_names = [f"clf_mnist-fcrelu_{i}.gz" for i in expansions]
            nn_module = ExpandableFcReLu_mnist
    elif ds == "cifar10":
        if network == "rff":
            model_folder = RF_CIFAR10
            clf_names = [f"clf_cifar10-rff_{i}.gz" for i in expansions]
            nn_module = RandomFourier
        elif network == "resnet":
            model_folder = RESN_CIFAR10
            clf_names = [f"clf_cifar10-resnet_{i}.gz" for i in expansions]
            nn_module = ResNet
    
    return model_folder, clf_names, nn_module

def get_pgd_attack_hyperparams(dataset_name):
    noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'

    lb, ub = 0, 1  # Bounds of the attack space. Can be set to `None` for
    # unbounded
    y_target = None  # None if `error-generic` or a class label for `error-specific`

    # Should be chosen depending on the optimization problem
    if dataset_name == 'mnist':
        solver_params = {
            'eta': 0.5,
            'eta_min': 0.1,
            'eta_max': None,
            'max_iter': 100,
            'eps': 1e-3
        }
    else:
        solver_params = {
            'eta': 0.1,
            'eta_min': 0.1,
            'eta_max': 0.1,
            'max_iter': 100,
            'eps': 1e-3
        }

    return solver_params, noise_type, lb, ub, y_target

def pretrain_secml(device, model_folder, clf_names, expansions, tr_dataset, nn_model, input_shape, output_classes, epoch, batch_size, lr):
    clfs = []
    # models are pre-trained with following number of training dataset:
    # mnist-fcrelu: 10k
    # mnist-cnn:
    # cifar10-resnet: 20k
    for name, n in zip(clf_names, expansions):
        if not(model_folder / name).exists():
            net = nn_model(expansion = n, out_classes = output_classes)
            print(f"Training network with {sum([i.numel() for i in list(net.parameters())])} parameters")
            net = net.to(device)
            net.train()
            clf = CClassifierPyTorch(net, input_shape=input_shape,
                                    optimizer=torch.optim.Adam(net.parameters(), lr=lr),
                                    loss=torch.nn.CrossEntropyLoss(), epochs=epoch, batch_size=batch_size)
            clf.fit(tr_dataset.X, tr_dataset.Y)
            print(clf)
            clf.model.eval()
            clf.save(str(model_folder / name))
            clfs.append(clf)
        else:
            clf = CClassifierPyTorch.load(str(model_folder / name))
            clf.model.eval()
            print(f"Loading network with {sum([i.numel() for i in list(clf.model.parameters())])} parameters")
            clfs.append(clf)
    return clfs
