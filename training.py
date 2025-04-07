import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from multiprocessing import Pool
import psutil
import numpy as np
import pandas as pd
from models import GATWithAutoencoderResidual

class RandomWalkMiniBatchGenerator:
    """
    Generator for mini-batches using random walks.
    """
    @staticmethod
    def perform_random_walk(node, edge_index, walk_length):
        visited_nodes = {node.item()}
        current_node = node
        for _ in range(walk_length):
            neighbors = edge_index[1][edge_index[0] == current_node]
            if len(neighbors) > 0:
                next_node = neighbors[torch.randint(len(neighbors), (1,))].item()
                visited_nodes.add(next_node)
                current_node = next_node
            else:
                break
        return visited_nodes

    @staticmethod
    def generate_mini_batches_with_walk_parallel(data, batch_size, walk_length=10, num_workers=4):
        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)
        for i in range(0, num_nodes, batch_size):
            batch_nodes = perm[i:i + batch_size]
            with Pool(processes=num_workers) as pool:
                results = pool.starmap(
                    RandomWalkMiniBatchGenerator.perform_random_walk,
                    [(node, data.edge_index, walk_length) for node in batch_nodes]
                )
            visited_nodes = set()
            for visited in results:
                visited_nodes.update(visited)
            visited_nodes = torch.tensor(list(visited_nodes), dtype=torch.long)
            batch_data = data.clone()
            batch_data.edge_index, batch_data.edge_attr = subgraph(
                visited_nodes, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=num_nodes
            )
            batch_data.x = data.x[visited_nodes]
            batch_data.y = data.y[visited_nodes]
            yield batch_data

class ModelEvaluator:
    """
    Evaluator for model performance using parallel mini-batches.
    """
    @staticmethod
    def evaluate_model_with_parallel(model, data, device, batch_size, walk_length):
        total_correct = 0
        total_samples = 0
        for batch_data in RandomWalkMiniBatchGenerator.generate_mini_batches_with_walk_parallel(data, batch_size, walk_length):
            output = model(batch_data.to(device))
            total_correct += (output.argmax(dim=1) == batch_data.y.to(device)).sum().item()
            total_samples += batch_data.y.size(0)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return accuracy

class GPUUsageEstimator:
    """
    Estimator for GPU and CPU RAM usage during training.
    """
    @staticmethod
    def estimate_total_gpu_ram_for_training(
        model, train_data, optimizer, device, batch_size, walk_length, num_epochs=1
    ):
        model.train()
        total_ram_usage = 0
        total_gpu_ram_usage = 0
        max_gpu_ram_usage = 0
        train_accuracies = []
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0
            process = psutil.Process()
            start_ram = process.memory_info().rss
            start_gpu_ram = torch.cuda.memory_allocated(device)
            batch_data = next(iter(RandomWalkMiniBatchGenerator.generate_mini_batches_with_walk_parallel(train_data, batch_size, walk_length)))
            optimizer.zero_grad()
            output = model(batch_data.to(device))
            loss = F.cross_entropy(output, batch_data.y.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, train_preds = output.max(dim=1)
            train_accuracy = (train_preds.cpu() == batch_data.y.cpu()).float().mean().item()
            train_accuracies.append(train_accuracy)
            end_ram = process.memory_info().rss
            ram_used = (end_ram - start_ram) / (1024 ** 2)
            end_gpu_ram = torch.cuda.memory_allocated(device)
            gpu_ram_used = (end_gpu_ram - start_gpu_ram) / (1024 ** 2)
            total_ram_usage += ram_used
            total_gpu_ram_usage += gpu_ram_used
            max_gpu_ram_usage = max(max_gpu_ram_usage, gpu_ram_used)
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"RAM Usage: {ram_used:.2f} MB, GPU RAM Usage: {gpu_ram_used:.2f} MB")
        avg_ram_usage = total_ram_usage / num_epochs
        avg_gpu_ram_usage = total_gpu_ram_usage / num_epochs
        print(f"Total GPU RAM Usage for {num_epochs} Epochs: {total_gpu_ram_usage:.2f} MB")
        print(f"Maximum GPU RAM Usage in a Single Epoch: {max_gpu_ram_usage:.2f} MB")
        print(f"Average GPU RAM Usage per Epoch: {avg_gpu_ram_usage:.2f} MB")
        return train_accuracies, total_gpu_ram_usage, max_gpu_ram_usage, avg_gpu_ram_usage