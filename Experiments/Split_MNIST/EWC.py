import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from torch import nn
from avalanche.models import IncrementalClassifier
from avalanche.models import BaseModel

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


class LwFCEPenalty(avl.training.LwF):
    """This wrapper around LwF computes the total loss
    by diminishing the cross-entropy contribution over time,
    as per the paper
    "Three scenarios for continual learning" by van de Ven et. al. (2018).
    https://arxiv.org/pdf/1904.07734.pdf
    The loss is L_tot = (1/n_exp_so_far) * L_cross_entropy +
                        alpha[current_exp] * L_distillation
    """
    def _before_backward(self, **kwargs):
        self.loss *= float(1/(self.clock.train_exp_counter+1))
        super()._before_backward(**kwargs)


def lwf_smnist(override_args=None):
    """
    "Learning without Forgetting" by Li et. al. (2016).
    http://arxiv.org/abs/1606.09282
    Since experimental setup of the paper is quite outdated and not
    easily reproducible, this experiment is based on
    "Three scenarios for continual learning" by van de Ven et. al. (2018).
    https://arxiv.org/pdf/1904.07734.pdf

    The hyper-parameter alpha controlling the regularization is increased over time, resulting
    in a regularization of  (1- 1/n_exp_so_far) * L_distillation
    """
    args = {'cuda': 0,
                                'lwf_alpha': [0, 0.5, 1.33333, 2.25, 3.2],
                                'lwf_temperature': 2, 'epochs': 21,
                                'layers': 1, 'hidden_size': 200,
                                'learning_rate': 0.001, 'train_mb_size': 128,
                                'seed': 1234}
    torch.manual_seed(args['seed'])
    device = torch.device(f"cuda:{args['cuda']}" if torch.cuda.is_available() and args['cuda'] >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitMNIST(5, return_task_id=False)
    model = MLP(hidden_size=args["hidden_size"], hidden_layers=args["layers"],
                initial_out_features=0, relu_act=False)
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    cl_strategy = LwFCEPenalty(
        model, SGD(model.parameters(), lr=args["learning_rate"]), criterion,
        alpha=args["lwf_alpha"], temperature=args["lwf_temperature"],
        train_mb_size=args["train_mb_size"], train_epochs=args["epochs"],
        device=device, evaluator=evaluation_plugin)

    res = None
    for experience in benchmark.train_stream:
        # cl_strategy.train(experience)
        print(experience.current_experience)
        # res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == '__main__':
    res = lwf_smnist()
    print(res)
