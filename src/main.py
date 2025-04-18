import os
import copy
import time
import torch
import pandas as pd
from torch import nn
from pathlib import Path
import torch.nn.functional as F
import torch.nn.utils.prune as tprune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

def clean_data_directory(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 1024)
        self.fc2 = torch.nn.Linear(1024, 2048)
        self.fc3 = torch.nn.Linear(2048, 2048)
        self.fc5 = torch.nn.Linear(2048, 27)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

def post_prune_model(model, amount):
    # Pruning
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            tprune.l1_unstructured(module, name='weight', amount=amount)

    #Remove the pruning reparametrizations to make the model explicitly sparse
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            tprune.remove(module, 'weight')


def get_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.EMNIST(root='dataset', split = 'letters', train=True, download=True, transform=transform)
    test_dataset = datasets.EMNIST(root='dataset', split = 'letters', train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def train():
    lr = 0.001
    weight_decay=1e-4
    epochs = 1
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()
    losses = []
    
    for _ in range(epochs):
        batch_losses = []
        for _, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.enable_grad():
                model.train()
                outputs = model(images)
                loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
        mean_epoch_loss = sum(batch_losses) / len(batch_losses)
        losses.append(mean_epoch_loss)
    return sum(losses) / len(losses)


def test_with_energy_measurement(model, sparsity, dataset, seed, quantization=False):
    csv_handler = CSVHandler(f'data/inference-energy-measurement-seeed-{seed}-sparsity-{sparsity}-quantization-{quantization}.csv')
    @measure_energy(handler=csv_handler)
    def test(model, sparsity, dataset, seed, quantization=False):
        criterion = nn.NLLLoss()
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        inference_time = pd.DataFrame(columns=['Batch','Time'])
        start_time = time.time()
        for batch_index, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        end_time = time.time()
        inference_time = inference_time._append(
            {'Batch': batch_index,'Time': end_time - start_time, 'Accuracy': correct / total},
            ignore_index=True
        )
        inference_time.to_csv(f'data/inference-time-seeed-{seed}-sparsity-{sparsity}-quantization-{quantization}.csv', index=False)
    test(model, sparsity, dataset, seed, quantization)
    csv_handler.save_data()


if __name__ == '__main__':

    train_dataset, test_dataset = get_dataset()
    device = 'cpu' 
    print(f'Using device: {device}')

    batch_size = 32
    max_seed = 1
    
    data_output_directory = Path('data')
    data_output_directory.mkdir(parents=True, exist_ok=True)
    model_output_directory = Path('model')
    model_output_directory.mkdir(parents=True, exist_ok=True)

    clean_data_directory('data')
    
    for seed in range(max_seed):
        print(f'Seed --- {seed}')
        model = Model()
        model.to(device)
        train()
        torch.save(model.state_dict(), f'model/model-seed-{seed}.pth')

        print(f'Sparsity --- 0.0 and no quantization')
        test_with_energy_measurement(model, 0.0, test_dataset, seed)

        quantization_dtypes = {
            'qint8': torch.qint8
        }

        for quantization_name, quantization_dtype in quantization_dtypes.items():
            model_q = torch.ao.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},  
                dtype=quantization_dtype
            )
            torch.save(model_q.state_dict(), f'model/model-{quantization_name}-seed-{seed}.pth')
            print(f'Sparsity --- 0.0 and quantization')
            test_with_energy_measurement(model, 0.0, test_dataset, seed, True)
        
        for sparsity in [0.3, 0.5, 0.7, 0.9]:
            print(f'Sparsity --- {sparsity}')
            sparse_model = Model()
            sparse_model.load_state_dict(copy.deepcopy(model.state_dict()))
            post_prune_model(sparse_model, sparsity)
            test_with_energy_measurement(sparse_model, sparsity, test_dataset, seed, False)