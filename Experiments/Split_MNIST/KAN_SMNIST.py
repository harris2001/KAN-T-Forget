import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../','../', 'Models')))
# from MLP import *
# from kan import KAN
from efficientKAN import *
# from Scaled_KAN import *
# from wavKAN import *
# from KANvolver import KANvolver as KAN

def kan_smnist(override_args=None):
    args = {'cuda': 0, 'hidden_size': 400,
            'hidden_layers': 2, 'epochs': 5, 'dropout': 0,
            'learning_rate': 0.001, 'train_mb_size': 16, 'seed': 1234}
    
    torch.manual_seed(args['seed'])
    device = torch.device(f"cuda:{args['cuda']}"
                          if torch.cuda.is_available() and
                          args['cuda'] >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitMNIST(5, return_task_id=False)

    # model = FastKAN(layers_hidden=[784,16,10], grid_min=4, grid_max=5, device=device)
    # model = KAN(layers_hidden=[784,16,10], grid_size=5, spline_order=5)
    # model = KAN(layers_hidden=[784,400,10])
    # model = KAN([28 * 28, 64, 10])
    model = KAN(layers_hidden=[784,16,10], grid_size=100, spline_order=4)
    model.to(device)

    criterion = CrossEntropyLoss()

    # interactive_logger = avl.logging.InteractiveLogger()
    # tensorboard_logger = avl.logging.TensorboardLogger('../results/tensorboard/gen_replay_pmnist_KAN')
    csv_logger = avl.logging.CSVLogger('../results/csv/gen_replay_pmnist_KAN')

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        metrics.forgetting_metrics(experience=True, stream=True),
        loggers=[csv_logger])

    # cl_strategy = avl.training.GenerativeReplay(
    #     model=model,
    #     optimizer=torch.optim.Adam(model.parameters(), lr=args['learning_rate']),
    #     criterion=criterion,
    #     train_mb_size=args['train_mb_size'],
    #     train_epochs=args['epochs'],
    #     eval_mb_size=128,
    #     replay_size=100,
    #     device=device,
    #     evaluator=evaluation_plugin,
    # )
    cl_strategy = avl.training.Naive(
        model=model,
        optimizer=SGD(model.parameters(), lr=args['learning_rate']),
        criterion=criterion,
        train_mb_size=args['train_mb_size'],
        train_epochs=args['epochs'],
        eval_mb_size=128,
        device=device,
        evaluator=evaluation_plugin,
    )

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = kan_smnist()
    print(res)