import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'Models')))

from Scaled_KAN import FastKAN


def get_conventional(hidden):
    # Create conventional MLP model
    class NN(nn.Module):

        def __init__(self):
            super(NN, self).__init__()
            self.fc1 = nn.Linear(28*28, hidden)
            self.fc2 = nn.Linear(hidden, 10)
            
        def forward(self, x):
            # Flatten the image
            x = x.view(-1, 28*28)
            # Pass through the first layer
            x = F.relu(self.fc1(x))
            # Pass through the second layer
            x = self.fc2(x)
            # Apply log softmax
            x = F.log_softmax(x, dim=1)
            return x
    return NN()

def main(hidden_conv, hidden_kan, grids, lr_conv, lr_kan, momentum_conv, momentum_kan, results_df):
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ###################
    # Hyperparameters #
    ###################

    batch_size_train = 32
    batch_size_test = 32

    tasks = 10


    print(device)
    model = get_conventional(hidden_conv)
    model.to(device)

    # Create KAN model
    KAN_model = FastKAN(layers_hidden=[28*28,hidden_kan,10], num_grids=grids, device=device)
    KAN_model.to(device)


    ###################
    # Data processing #
    ###################

    # Transform the MNIST dataset to a tensor and normalising it
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)), # Normalise samples using the mean of 0.1307 and std of 0.3081 - extracted after manual analysis
        T.Lambda(lambda x: torch.flatten(x))
    ])
    train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('../data', train=False, download=True, transform=transform)

    # optimizer = optim.Adam(model.parameters(), lr=lr_conv)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=momentum_conv)
    # loss_fn = nn.CrossEntropyLoss()


    optimizer_KAN = optim.Adam(KAN_model.parameters(), lr=lr_kan)
    scheduler_KAN = optim.lr_scheduler.ExponentialLR(optimizer_KAN, gamma=momentum_kan)
    loss_fn_KAN = nn.CrossEntropyLoss()

    # Pair numbers together and create 5 tasks
    tasks = [(i,i+1) for i in range(0,10,2)]

    # For training: fill splitted_train array with examples where the target value is in task i
    targets = train_set.targets.numpy()
    splitted_train = []
    for task in tasks:
        indices = np.where(np.isin(targets, task))[0]
        subset = [train_set[i] for i in indices]
        splitted_train.append(subset)

    # For testing: fill splitted_train array with examples where the target value is in task i
    targets = test_set.targets.numpy()
    splitted_test = []
    for task in tasks:
        indices = np.where(np.isin(targets, task))[0]
        subset = [test_set[i] for i in indices]
        splitted_test.append(subset)


    ###################
    #    Training     #
    ###################

    # Train the KAN
    performance = []
    train_accs_conv = []
    test_accs_conv = []
    train_accs_kan = []
    test_accs_kan = []

    for task in range(len(tasks)):
        print(f"Training on {task}")

        # Create dataloader with the splitted training and testing sets
        train_loader = torch.utils.data.DataLoader(
            splitted_train[task],
            batch_size=batch_size_train,
            shuffle=True
        )

        # Accumulate testing samples into performance array to evaluate forgetting of previous tasks
        performance.extend(splitted_test[task])
        
        test_loader = torch.utils.data.DataLoader(
            performance, 
            batch_size=batch_size_test, 
            shuffle=True
        )

        print(len(train_loader.dataset), len(test_loader.dataset))
        
        for epoch in range(15):
            # Train the conventional model
            # model.train()
            # correct = 0

            # for data, target in train_loader:
            #     data, target = data.to(device), target.to(device)
            #     optimizer.zero_grad()
            #     output = model(data)
            #     loss = loss_fn(output, target)
            #     loss.backward()
            #     optimizer.step()

            #     pred = output.argmax(dim=1, keepdim=True)
            #     correct += pred.eq(target.view_as(pred)).sum().item()

            # train_acc_conv = correct / len(train_loader.dataset)
            
            # Train the KAN model
            KAN_model.train()
            correct = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer_KAN.zero_grad()
                output = KAN_model(data)
                loss = loss_fn_KAN(output, target)
                loss.backward()
                optimizer_KAN.step()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            train_acc_kan = correct / len(train_loader.dataset)

            train_acc = [100, train_acc_kan]

            KAN_model.eval()
            model.eval()
            test_loss = [0,0]
            correct = [0,0]

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = [model(data),KAN_model(data)]
                    # test_loss[0] += loss_fn(output[0], target).item() * data.size(0)
                    test_loss[1] += loss_fn_KAN(output[1], target).item() * data.size(0)
                    pred = [output[0].argmax(dim=1, keepdim=True), output[1].argmax(dim=1, keepdim=True)]
                    correct[0] += pred[0].eq(target.view_as(pred[0])).sum().item()
                    correct[1] += pred[1].eq(target.view_as(pred[1])).sum().item()
            test_loss[0] /= len(test_loader.dataset)
            test_loss[1] /= len(test_loader.dataset)
            
            test_acc = [0,0]
            test_acc[0] = correct[0] / len(test_loader.dataset)
            test_acc[1] = correct[1] / len(test_loader.dataset)

            # Update the learning rate
            scheduler_KAN.step()
            # print training and testing accuracy for each model
            print(f"Epoch {epoch}:")
            print(f"Conventional model: Train accuracy: {train_acc[0]*100:.2f}, Test accuracy: {test_acc[0]*100:.2f}    VS   KAN model: Train accuracy: {train_acc[1]*100:.2f}, Test accuracy: {test_acc[1]*100:.2f}")

            # Store results
            new_row = pd.DataFrame({
                'Task': [task],
                'Epoch': [epoch],
                'LR_Conv': [lr_conv],
                'LR_KAN': [lr_kan],
                'Momentum_Conv': [momentum_conv],
                'Momentum_KAN': [momentum_kan],
                'Train_Acc_Conv': [train_acc[0]],
                'Test_Acc_Conv': [test_acc[0]],
                'Train_Acc_KAN': [train_acc[1]],
                'Test_Acc_KAN': [test_acc[1]]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)

            # Save results to CSV
            new_row.to_csv('training_results.csv', mode='a', header=not os.path.exists('training_results.csv'), index=False)
            
        # Store accuracies for plotting
        train_accs_conv.append(train_acc[0])
        test_accs_conv.append(test_acc[0])
        train_accs_kan.append(train_acc[1])
        test_accs_kan.append(test_acc[1])

    ###################
    #   Evaluation    #
    ###################

    # Plot accuracies
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(train_accs_conv, label='Conventional Train Accuracy')
    axs[0].plot(train_accs_kan, label='KAN Train Accuracy')
    axs[0].set_title('Train Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(test_accs_conv, label='Conventional Test Accuracy')
    axs[1].plot(test_accs_kan, label='KAN Test Accuracy')
    axs[1].set_title('Test Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Save the plot with a descriptive name
    plot_filename = f"plot_lr_conv_{lr_conv}_lr_kan_{lr_kan}_mom_conv_{momentum_conv}_mom_kan_{momentum_kan}.png"
    plt.savefig(plot_filename)
    # plt.show()

    # torch.save(model.state_dict(), "mnist_model.pth")

if __name__ == "__main__":

    ###################
    # Hyperparameters #
    ###################

    lr_conv = [0.0001]
    lr_kan  = [0.00001, 0.000001, 0.0000001]
    momentum_conv = [0.8]
    momentum_kan = [0.7,0.9]
    hidden_conv = 2**7
    hidden_kan = 2**5
    grids = 4

    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['Task', 'Epoch', 'LR_Conv', 'LR_KAN', 'Momentum_Conv', 'Momentum_KAN', 'Train_Acc_Conv', 'Test_Acc_Conv', 'Train_Acc_KAN', 'Test_Acc_KAN'])

    # Run the experiments
    for lr in lr_conv:
        for lr_k in lr_kan:
            for mom in momentum_conv:
                for mom_k in momentum_kan:
                    main(hidden_conv, hidden_kan, grids, lr, lr_k, mom, mom_k, results_df)

    print("Results saved to training_results.csv")
