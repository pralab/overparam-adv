import torch
from torch import nn
import numpy as np
import argparse
import time
from autoattack import AutoAttack
from secml.adv.attacks import CFoolboxPGDL2, CAttackEvasionPGDLS
from secml.adv.seceval import CSecEval, CSecEvalData
from secml.array import CArray
from utils.data_utils import load_pytorch_dataset, pytorch_ds_to_secml_ds
from utils.model_utils import get_models_and_path, pretrain_secml, get_pgd_attack_hyperparams
from utils.plot_utils import plot_base_performance, plot_robustness_performance, plot_base_performance_ambrastyle
import utils.config as config
from folder import PGDL2_CNN_MNIST, AA_CNN_MNIST, PGDL2_FCRELU_MNIST, AA_FCRELU_MNIST, PGDL2_RFF_CIFAR10, AA_RFF_CIFAR10, PGDL2_RESN_CIFAR10, AA_RESN_CIFAR10, PLOT_FOLDER

"""
Sample command python run.py --ds mnist --model cnn --attack pgdl2 --train 20
"""

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def runPGDL2attack_secml(attack, parameters, clfs, tr_dataset, ts_dataset, folder_pretrained_model, epsilons, solver_params, dist, lb, ub, y_target):
    for p, clf in zip(parameters, clfs):

        if not (folder_pretrained_model/f"security_evaluation_{p}.gz").exists():

            pgd_l2_attack = CAttackEvasionPGDLS(classifier=clf, 
                                                double_init_ds=tr_dataset,
                                                double_init=True,
                                                y_target=y_target, 
                                                distance=dist,
                                                dmax=0,
                                                lb = lb, ub = ub,
                                                solver_params=solver_params)
            pgd_l2_attack.verbose = 0
            sec_eval_pgd = CSecEval(pgd_l2_attack, "dmax", epsilons, save_adv_ds=False)
            
            print("[INFO] Attack Starterd ...")
            starting_time = time.time()
            sec_eval_pgd.run_sec_eval(ts_dataset)
            end_time = time.time()
            print(f"Time taken to run security evaluation for {attack} attack: {end_time-starting_time}")
            
            sec_eval_pgd.sec_eval_data.save(folder_pretrained_model/f"security_evaluation_{p}.gz")
        else:
            print(f"Security evaluation for {p} param model already computed. All set for robustness plots")
    

def runAutoAttack_pytorch(attack, parameters, clfs, sec_eval_folder, epsilons, test_loader, batch_size = 100):
    print("batchsize: ", batch_size)
    l = [x for (x, y) in test_loader]   
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    
    for p, clf in zip(parameters, clfs):
        if not (sec_eval_folder/f"security_evaluation_{p}.gz").exists():
            csecevaldata_obj =  CSecEvalData()
            csecevaldata_obj.param_name = "epsilons"
            csecevaldata_obj.param_values = epsilons
            csecevaldata_obj.Y = CArray(y_test.numpy()).deepcopy()  # original labels
            csecevaldata_obj.Y_pred =  []
            csecevaldata_obj.scores = []
            csecevaldata_obj.time = []
            nnet = clf.model

            att_pred = []
            cum_att_scores = []
            for eps in epsilons:
                AA = AutoAttack(nnet, norm="L2", eps=eps, seed=SEED, version="standard", device=device)
                print(f"\nStart autoattack for epsilon: {eps}")
                start_time = time.time()
                x_adv, y_adv = AA.run_standard_evaluation(x_test, y_test, batch_size, return_labels=True)
                # att_scores = nnet(x_adv)
                # cum_att_scores.append(att_scores)
                # pred_proba = nn.Softmax(dim=1)(att_scores)
                # att_pred.append(CArray(pred_proba.argmax(1)))  
                att_pred.append(CArray(y_adv))
                end_time = time.time()
                
            
            csecevaldata_obj.scores.append(cum_att_scores)
            csecevaldata_obj.Y_pred.append(att_pred)
            csecevaldata_obj.time.append(end_time- start_time)

            csecevaldata_obj.save(sec_eval_folder/f"security_evaluation_{p}.gz")
        else:
            print(f"Security evaluation for {p} param model already computed. All set for robustness plots")


if __name__ == "__main__":

    SEED = 121

    parser = argparse.ArgumentParser(description="Run multiple trainings with growing network capacity")
    parser.add_argument("--ds", type=str, help="select dataset : mnist or cifar10",
        choices={"mnist", "cifar10"}, dest="dataset_name", default="mnist")
    parser.add_argument(
        "--model", type=str, help="select model to train: cnn or fcrelu", choices={"cnn", "fcrelu", "rff", "resnet"}, dest="model", default="cnn")
    parser.add_argument(
        "--epochs", type=int, help="The number of epochs to train, if not given, the value in the config file is taken", dest="epochs")
    parser.add_argument(
        "--attack", type=str, help="choose the type of attack: pgdl2, autoattck", choices={"pgdl2", "autoattack"}, dest="attack_name", default="pgdl2")
    parser.add_argument(
        "--train", type=int, help="give the size of the training set", dest="train_subset")

    
    args = parser.parse_args()
    print(f"seed: {SEED}")

    ds_name, model_name, attack, train_subset = args.dataset_name, args.model, args.attack_name, args.train_subset
    lr, steps, stepsize = config.LR1, config.STEPS, config.STEPSIZE
    # batch_size needs to me a number that is a factor of train and test subset number, BATCHSIZE3 = 20
    
    # epochs = args.epochs if args.epochs else config.EPOCH1
    if args.epochs:
        epochs = args.epochs
    else:
        if ds_name == "mnist":
            epochs = config.EPOCH1
        elif ds_name == "cifar10":
            epochs = config.EPOCH3  # changed for rerun 18/01

    if ds_name == "mnist":
        input_shape = config.MNIST_INPUT_SHAPE
        out_classes = config.MNIST_OUTCLASSES
        test_subset = config.MNIST_TESTSIZE
        batch_size = config.BATCHSIZE3
        ds_normalization = False
        # solver_params, noise_type, lb, ub, y_target = get_pgd_attack_hyperparams(ds_name)
        xscale_base = 2
        if model_name == "cnn":
            epsilons = np.linspace(0.3, 3.0, 5)
            expansions = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30]
            train_subset = config.MNIST_TRAINSIZE
            fig_title_base_perf = "Performance on benign samples MNIST: CNN"
            if attack == "pgdl2":
                sec_eval_folder = PGDL2_CNN_MNIST
            elif attack == "autoattack":
                sec_eval_folder = AA_CNN_MNIST
        elif model_name == "fcrelu":
            epsilons = np.linspace(0.03, 2.5, 5)
            expansions = [4, 6, 8, 10, 15, 20, 25, 30, 35, 40]
            train_subset = config.MNIST_TRAINSIZE
            fig_title_base_perf = "Performance on benign samples MNIST: FCRELU"
            if attack == "pgdl2":
                sec_eval_folder = PGDL2_FCRELU_MNIST
            elif attack == "autoattack":
                sec_eval_folder = AA_FCRELU_MNIST
    elif ds_name == "cifar10":
        input_shape = config.CIFAR10_INPUT_SHAPE
        out_classes = config.CIFAR10_OUTCLASSES
        test_subset = config.CIFAR10_TESTSIZE
        batch_size = config.BATCHSIZE5
        ds_normalization = False
        xscale_base = 5
        if model_name == "rff":
            epsilons = np.linspace(0.01, 0.3, 5)
            expansions = [4, 6, 8, 10, 15, 20, 25, 30, 35, 40]
            fig_title_base_perf = "Performance on benign samples CIFAR10: Random Fourier Features"
            if attack == "pgdl2":
                sec_eval_folder = PGDL2_RFF_CIFAR10
            elif attack == "autoattack":
                sec_eval_folder = AA_RFF_CIFAR10
        elif model_name == "resnet":
            epsilons = np.linspace(0.01, 0.3, 5)
            expansions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            # expansions = [-5, -2, 0, 1, 4, 5, 6, 7, 8, 9]
            train_subset = config.CIFAR10_TRAINSIZE
            fig_title_base_perf = "Performance on benign samples CIFAR10: ResNet"
            if attack == "pgdl2":
                sec_eval_folder = PGDL2_RESN_CIFAR10
            elif attack == "autoattack":
                sec_eval_folder = AA_RESN_CIFAR10

        # add specification when doing cifar10 experiments
    
    print(f"[INFO] Loading {ds_name} dataset in pytorch ...")
    dataset = load_pytorch_dataset(SEED, ds_name, train_subset, test_subset, batch_size, ds_normalization)
    print(f"[INFO] loading {model_name} model ...")
    model_folder, clf_names, model = get_models_and_path(ds_name, model_name, expansions)
    print("[INFO] Converting pytorch dataset to secml format...")
    tr_secml = pytorch_ds_to_secml_ds(dataset.train_loader, batch_size)
    ts_secml = pytorch_ds_to_secml_ds(dataset.test_loader, batch_size)
    print("ts_secml: ", tr_secml.X.shape)
    print("ts_secml: ", ts_secml.X.shape)
    print("[INFO] Pretraining/Loading existing network ...")

    clfs = pretrain_secml(device, model_folder, clf_names, expansions, tr_secml, model, input_shape, out_classes, epochs, batch_size, lr)
    
    # if not (PLOT_FOLDER /f"base_perf_{ds_name}-{model_name}.pdf").exists():
    print(f"[INFO] Plotting base performance for {ds_name}-{model_name} ...")
    tr_acc, ts_acc = plot_base_performance_ambrastyle(clfs, tr_secml, ts_secml, ds_name, model_name, fig_title_base_perf, train_subset, xscale_base)
    
    parameters = [sum([i.numel() for i in list(clf.model.parameters())]) for clf in clfs]
    solver_params, noise_type, lb, ub, y_target = get_pgd_attack_hyperparams(ds_name)

    print("[INFO] Attacking the model begins ...")

    if attack == "pgdl2":
        runPGDL2attack_secml(attack, parameters, clfs, tr_secml, ts_secml, sec_eval_folder, epsilons, solver_params, noise_type, lb, ub, y_target)
        print(f"\n[INFO] Plotting security evaluation plots {ds_name}-{model} - {attack}...")
        plot_robustness_performance(ds_name, model_name, attack, parameters, sec_eval_folder, ts_acc)
    elif attack == "autoattack":
        runAutoAttack_pytorch(attack, parameters, clfs, sec_eval_folder, epsilons, dataset.test_loader)
        print(f"\n[INFO] Plotting security evaluation plots {ds_name}-{model} - {attack}...")
        plot_robustness_performance(ds_name, model_name, attack, parameters, sec_eval_folder, ts_acc)

    
    
    



## TO DO
## Add test_subset size  -- DONE
## check the functiont hat converts pytorch datasetr to secml   -- DONE
## in pgdl2 attack, try with CAttackEvasionPGDLS insteaf of CFoolboxPGDL2  -- DONE
## in autoattack, the expected inbut is eith 3D or 4D, currently its in the shape of (batch_size, img_size): [20, 784] - check the loader function and load in orig pytorch format without reshaping and check whether the func pytorch_ds_to_secml is able to correctly change the datashape else use reshape -- DONE

## Add multiple seeds to experiments:
        # 1. have a fixed set of 5 seeds
        # need to seed the test loader to add variation in the testing set
            # in load_pytorch_dataset() change sampler for test dataset
            # in runAutoAttack_pytorch() run for loop once to get x_test and y_test together since the test-loader is not fixed now
        # 2. change plot_base_performance_ambrastyle() function to get vectors of tr_acc and ts_acc for each seed and then plot base performance along with standard deviation
        # 3. change plot_robustness_performance() by creating a vector of full_range_acc() for each seed and the plot robust along with its standard deviation
