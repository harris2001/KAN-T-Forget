import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from torch import nn
from avalanche.models import IncrementalClassifier
from avalanche.models import BaseModel

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../','../', 'Models')))
from Scaled_KAN import *

class MLP(nn.Module, BaseModel):
    def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
                 output_size=10, drop_rate=0, relu_act=True, initial_out_features=0):
        """
        :param initial_out_features: if >0 override output size and build an
            IncrementalClassifier with `initial_out_features` units as first.
        """
        super().__init__()
        self._input_size = input_size

        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                                 nn.Dropout(p=drop_rate)))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)

        if initial_out_features > 0:
            self.classifier = IncrementalClassifier(in_features=hidden_size,
                                                    initial_out_features=initial_out_features)
        else:
            self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        return self.features(x)

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
    #             drop_rate=args['dropout'])
    model = FastKAN(layers_hidden=[28*28,2**5,10], num_grids=4, grid_min=2, grid_max=4, device=device)
    model.to(device)
    
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    cl_strategy = avl.training.EWC(
        model, SGD(model.parameters(), lr=args['learning_rate']), criterion,
        ewc_lambda=args['ewc_lambda'], mode=args['ewc_mode'], decay_factor=args['ewc_decay'],
        train_mb_size=args['train_mb_size'], train_epochs=args['epochs'], eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = ewc_pmnist()
    print(res)