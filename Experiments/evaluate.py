import torch
import sys
import os
import avalanche as avl

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../', 'Models')))
from LAMAML import MTConvCIFAR

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'saved_models')))

from load_model import load

model = MTConvCIFAR()
model.load_state_dict(load('lamaml.pth'))
model.to('cuda')
model.eval()


benchmark = avl.benchmarks.SplitCIFAR100(n_experiences=20,
                                            return_task_id=True)



for experience in benchmark.test_stream:
    acc = 0
    for i, (x, y, t) in enumerate(experience.dataset):
        x = x.to('cuda')
        print(x.shape, y, t)
        y_hat = model(x, t)
        acc += (y_hat.argmax(1) == y).float().mean().item()
    print(f'Accuracy: {acc/len(experience.dataset)}')