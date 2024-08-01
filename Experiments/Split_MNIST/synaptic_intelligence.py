import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from avalanche.evaluation import metrics as metrics
# from models import MultiHeadMLP

import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../', '../', 'Utils')))
from utils import set_seed, create_default_args

import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../', '../', 'Models')))
# from Scaled_KAN import *

from kan import KAN

def synaptic_intelligence_smnist(override_args=None):
    """
    "Continual Learning Through Synaptic Intelligence" by Zenke et. al. (2017).
    http://proceedings.mlr.press/v70/zenke17a.html
    """
    args = create_default_args({'cuda': 0, 'si_lambda': 1, 'si_eps': 0.1, 'epochs': 5,
                                'learning_rate': 0.0001, 'train_mb_size': 64, 'seed': None},
                               override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")
    benchmark = avl.benchmarks.SplitMNIST(5, return_task_id=True,
                                          fixed_class_order=list(range(10)))
    
    # model = MultiHeadMLP(hidden_size=256, hidden_layers=2)
    # model = FastKAN(layers_hidden=[28*28,256,10], num_grids=8, device=device)
    model = KAN(width=[28*28,256,10], grid=3, k=3, seed=0).to(device)

    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    cl_strategy = avl.training.SynapticIntelligence(
        model=model, optimizer=Adam(model.parameters(), lr=args.learning_rate), criterion=criterion,
        si_lambda=args.si_lambda, eps=args.si_eps,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    dataset = {}
    for experience in benchmark.train_stream:
        train_input = []
        train_label = []
        for (x,y,t) in experience.dataset:
            train_input.append(x)
            train_label.append(y)
        dataset['train_input'] = train_input
        dataset['train_label'] = train_label
        model.fit(dataset, opt = 'LBFGS', steps=10, update_grid=True)
        # model.train(dataset=dataset, epochs=10, batch_size=64, opt="Adam", lr=0.001)
    for experience in benchmark.test_stream:
        test_input = []
        test_label = []
        for (x,y,t) in experience.dataset:
            test_input.append(x)
            test_label.append(y)
        dataset['test_input'] = test_input
        dataset['test_label'] = test_label

    res = cl_strategy.eval(benchmark.test_stream)
    res = 0
    return res


if __name__ == '__main__':
    res = synaptic_intelligence_smnist()
    print(res)