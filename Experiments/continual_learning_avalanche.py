import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    labels_repartition_metrics,
    loss_metrics,
    MAC_metrics,
    bwt_metrics,
    forward_transfer_metrics,
    class_accuracy_metrics,
    amca_metrics,
    ExperienceForgetting,
)
from avalanche.models import SimpleMLP
from avalanche.logging import (
    TextLogger,
    CSVLogger,
    TensorboardLogger,
)
from avalanche.training.plugins import EvaluationPlugin, EWCPlugin, ReplayPlugin
from avalanche.training.supervised import EWC,Naive

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'../','Models')))
from Scaled_KAN import FastKAN

def main(args):
    # Choose GPU to run experiment on
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    # Prepare the data
    if args.dataset == "mnist":
        # Transform dataset
        train_transform = transforms.Compose(
            [
                RandomCrop(28, padding=4),
                ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)) # Normalise MNIST dataset
            ]
        )
        test_transform = transforms.Compose(
            [ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        
        mnist_train = MNIST(
            root=default_dataset_location("mnist"),
            train=True,
            download=True,
            transform=train_transform,
        )
        mnist_test = MNIST(
            root=default_dataset_location("mnist"),
            train=False,
            download=True,
            transform=test_transform,
        )
        benchmark = nc_benchmark(
            train_dataset=mnist_train, test_dataset=mnist_test, n_experiences=5,
            task_labels=False, seed=args.seed
        )
    elif args.dataset == "cifar10":
        
        train_transform = transforms.Compose([
            ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)), # Normalise CIFAR10 dataset
        ])
        test_transform = transforms.Compose([
            ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])

        cifar_train = CIFAR10(
            root=default_dataset_location("cifar10"),
            train=True,
            download=True,
            transform=train_transform,
        )
        cifar_test = CIFAR10(
            root=default_dataset_location("cifar10"),
            train=False,
            download=True,
            transform=test_transform,
        )
        benchmark = nc_benchmark(
            train_dataset=cifar_train, test_dataset=cifar_test, n_experiences=20,
            task_labels=False, seed=args.seed
        )

    # MODEL SELECTION
    if args.model == "SimpleMLP":
        if args.dataset == "mnist":
            model = SimpleMLP(
                num_classes=benchmark.n_classes,
                input_size=28 * 28,
                hidden_size=args.hidden,
                hidden_layers=1
            )
        elif args.dataset == "cifar10":
            model = SimpleMLP(
                num_classes=benchmark.n_classes,
                input_size=32*32*3,
                hidden_size=args.hidden,
                hidden_layers=1
            )
    elif args.model =="FastKAN":
        if args.dataset == "mnist":
            model = FastKAN(
                layers_hidden=[784,args.hidden,10],
                num_grids=20,
                device=device
            )
        elif args.dataset == "cifar10":
            model = FastKAN(
                layers_hidden=[32*32*3,args.hidden,128,100],
                num_grids=4,
                device=device
            )

    # log to text file
    text_logger = TextLogger(open("results/log.txt", "a"))

    csv_logger = CSVLogger(log_folder='results/csv/results_'+args.dataset+"_"+args.model+"_"+str(args.hidden)+"_"+str(args.rnd))

    tb_logger = TensorboardLogger(tb_log_dir='results/tensorboard/tb_log_'+args.dataset+"_"+args.model+"_"+str(args.hidden)+"_"+str(args.rnd))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True,
            stream=True,
        ),
        loss_metrics(
            epoch=True,
            stream=True,
        ),
        ExperienceForgetting(),
        class_accuracy_metrics(
            epoch=True, stream=True, classes=list(range(benchmark.n_classes))
        ),
        amca_metrics(),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        forward_transfer_metrics(experience=True, stream=True),
        MAC_metrics(minibatch=True, epoch=True, experience=True),
        labels_repartition_metrics(on_train=True, on_eval=True),
        loggers=[text_logger, csv_logger, tb_logger],
        collect_all=True,
    )  # collect all metrics (set to True by default)width

    ewc = EWCPlugin(ewc_lambda=1)
    replay = ReplayPlugin(mem_size=2000)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model=model,
        optimizer=Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999)),
        criterion=CrossEntropyLoss(),
        train_mb_size=500,
        train_epochs=args.epochs,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin,
        plugins=[ewc],
        eval_every=args.epochs
    )
    
    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for i, experience in enumerate(benchmark.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        res = cl_strategy.train(experiences=experience, eval_streams=[benchmark.test_stream])
        print("Training completed")

        print("Computing accuracy on the whole test set")
        # test returns a dictionary with the last metric collected during
        # evaluation on that stream
        results.append(cl_strategy.eval(benchmark.test_stream))
        break
    print(f"Test metrics:\n{results}")

    # Dict with all the metric curves,
    # only available when `collect_all` is True.
    # Each entry is a (x, metric value) tuple.
    # You can use this dictionary to manipulate the
    # metrics without avalanche.
    all_metrics = cl_strategy.evaluator.get_all_metrics()
    print(f"Stored metrics: {list(all_metrics.keys())}")
    torch.save(model.state_dict(), "saved_models/continual_"+args.dataset+"_"+args.model+"_"+str(args.hidden)+"_"+str(args.rnd)+".pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--model", type=str, default="SimpleMLP", help="Select model to use: SimpleMLP or FastKAN")
    parser.add_argument("--dataset", type=str, default="mnist", help="Select dataset to use: mnist or cifar10")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--rnd", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    main(args)