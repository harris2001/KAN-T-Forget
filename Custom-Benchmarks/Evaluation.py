import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    labels_repartition_metrics,
    loss_metrics,
    cpu_usage_metrics,
    timing_metrics,
    gpu_usage_metrics,
    ram_usage_metrics,
    disk_usage_metrics,
    MAC_metrics,
    bwt_metrics,
    forward_transfer_metrics,
    class_accuracy_metrics,
    amca_metrics,
)
from avalanche.models import SimpleMLP
from avalanche.logging import (
    TextLogger,
    CSVLogger,
    TensorboardLogger,
)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive

from Scaled_KAN import *

def main(args):
    # Choose GPU to run experiment on
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

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
    
    # Prepare the data
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
        task_labels=False, seed=1234
    )

    # MODEL CREATION
    model = SimpleMLP(num_classes=benchmark.n_classes)
    print(model)
    # model = FastKAN(layers_hidden=[784,32,10], num_grids=4, device=device)
    # print(model)
    # DEFINE THE EVALUATION PLUGIN AND LOGGER
    # The evaluation plugin manages the metrics computation.
    # It takes as argument a list of metrics and a list of loggers.
    # The evaluation plugin calls the loggers to serialize the metrics
    # and save them in persistent memory or print them in the standard output.

    # log to text file
    text_logger = TextLogger(open("log.txt", "a"))

    csv_logger = CSVLogger()

    tb_logger = TensorboardLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        loss_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        class_accuracy_metrics(
            epoch=True, stream=True, classes=list(range(benchmark.n_classes))
        ),
        amca_metrics(),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        forward_transfer_metrics(experience=True, stream=True),
        cpu_usage_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        timing_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        ram_usage_metrics(
            every=0.5, minibatch=True, epoch=True, experience=True, stream=True
        ),
        gpu_usage_metrics(
            args.cuda,
            every=0.5,
            minibatch=True,
            epoch=True,
            experience=True,
            stream=True,
        ),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        MAC_metrics(minibatch=True, epoch=True, experience=True),
        labels_repartition_metrics(on_train=True, on_eval=True),
        loggers=[text_logger, csv_logger, tb_logger],
        collect_all=True,
    )  # collect all metrics (set to True by default)
    
    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model=model,
        optimizer=Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999)),
        criterion=CrossEntropyLoss(),
        train_mb_size=500,
        train_epochs=1,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin,
        eval_every=1,
    )
    
    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for i, experience in enumerate(benchmark.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary containing last recorded value
        # for each metric.
        print(experience.classes_seen_so_far)
        res = cl_strategy.train(experiences=experience, eval_streams=[benchmark.test_stream])
        print("Training completed")

        print("Computing accuracy on the whole test set")
        # test returns a dictionary with the last metric collected during
        # evaluation on that stream
        results.append(cl_strategy.eval(benchmark.test_stream))

    print(f"Test metrics:\n{results}")

    # Dict with all the metric curves,
    # only available when `collect_all` is True.
    # Each entry is a (x, metric value) tuple.
    # You can use this dictionary to manipulate the
    # metrics without avalanche.
    all_metrics = cl_strategy.evaluator.get_all_metrics()
    print(f"Stored metrics: {list(all_metrics.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)