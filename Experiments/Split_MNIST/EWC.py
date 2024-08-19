import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../','../', 'Models')))
from Scaled_KAN import *
# from torchKAN import *
# from wavKAN import *
# from KANvolver import *

def ewc_pmnist(override_args=None):
    """
    "Overcoming catastrophic forgetting in neural networks" by Kirkpatrick et. al. (2017).
    https://www.pnas.org/content/114/13/3521

    Results are below the original paper, which scores around 94%.
    """
    args = {'cuda': 0, 'ewc_lambda': 1, 'hidden_size': 512,
                                'hidden_layers': 1, 'epochs': 10, 'dropout': 0,
                                'ewc_mode': 'separate', 'ewc_decay': None,
                                'learning_rate': 0.001, 'train_mb_size': 256,
                                'seed': 1234}
    torch.manual_seed(args['seed'])
    device = torch.device(f"cuda:{args['cuda']}"
                          if torch.cuda.is_available() and
                          args['cuda'] >= 0 else "cpu")

    benchmark = avl.benchmarks.PermutedMNIST(10)
    # model = MLP(hidden_size=args['hidden_size'], hidden_layers=args['hidden_layers'],
    #             drop_rate=args['dropout']) # 668672 params
    model = FastKAN(layers_hidden=[28*28,10], grid_min=100, grid_max=101, device=device) # 406528 params
    # model = KAN(layers_hidden=[784,32,10])
    # model = KANvolver(layers_hidden=[32, 10], polynomial_order=3)
    model.to(device)
    
    criterion = CrossEntropyLoss()

    # interactive_logger = avl.logging.InteractiveLogger()
    # tensorboard_logger = avl.logging.TensorboardLogger('../results/tensorboard/ewc_pmnist_KAN_2')
    csv_logger = avl.logging.CSVLogger('../results/csv/ewc_pmnist_KAN_2')

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        metrics.forgetting_metrics(experience=True, stream=True),
        loggers=[csv_logger])

    cl_strategy = avl.training.EWC(
        model, SGD(model.parameters(), lr=args['learning_rate']), criterion,
        ewc_lambda=args['ewc_lambda'], mode=args['ewc_mode'], decay_factor=args['ewc_decay'],
        train_mb_size=args['train_mb_size'], train_epochs=args['epochs'], eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)
    # cl_strategy = avl.training.Naive(model, SGD(model.parameters(), lr=args['learning_rate']), criterion,
    #                                  train_mb_size=args['train_mb_size'], train_epochs=args['epochs'],
    #                                  eval_mb_size=128, device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = ewc_pmnist()
    print(res)