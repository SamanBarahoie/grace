import argparse
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from models import GATWithAutoencoderResidual
from utils import GraphLoader, EmbeddingsVisualizer
from training import GPUUsageEstimator
import config

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a GAT model with autoencoder and residual connections.")
    parser.add_argument('--data-path', type=str, default=config.DATA_PATH, help="Path to the .npz graph data file")
    parser.add_argument('--walk-length', type=int, default=config.WALK_LENGTH, help="Length of random walk")
    parser.add_argument('--hidden-dim', type=int, default=config.HIDDEN_DIM, help="Hidden layer dimension")
    parser.add_argument('--encoding-dim', type=int, default=config.ENCODING_DIM, help="Encoding dimension")
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=config.BATCH_SIZES, help="List of batch sizes")
    parser.add_argument('--output-csv', type=str, default=config.OUTPUT_CSV, help="Output CSV file for accuracy results")
    parser.add_argument('--device', type=str, default=config.DEVICE, help="Device to use (cuda/cpu)")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load data
    loader = GraphLoader(args.data_path)
    graph, labels = loader.load_graph()
    train_data = loader.to_pyg_data()

    # Set dynamic hyperparameters
    input_dim = train_data.x.shape[1]
    output_dim = len(np.unique(train_data.y))
    device = torch.device(args.device)

    # Training setup
    accuracy_results = []
    total_gpu_usage = []
    max_gpu_usage = []
    avg_gpu_usage = []
    final_accuracy_us = []

    for batch_size in args.batch_sizes:
        print(f"\nTraining with batch size: {batch_size}\n")
        model = GATWithAutoencoderResidual(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            encoding_dim=args.encoding_dim
        )
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Train and estimate GPU usage
        train_accuracies, total_gpu_ram_usage, max_gpu_ram_usage, avg_gpu_ram_usage = GPUUsageEstimator.estimate_total_gpu_ram_for_training(
            model, train_data, optimizer, device, batch_size, args.walk_length, args.epochs
        )

        # Record results
        final_accuracy = train_accuracies[-1]
        final_accuracy_us.append(final_accuracy)
        accuracy_results.append({'Batch Size': batch_size, 'Final Accuracy': final_accuracy})
        total_gpu_usage.append(total_gpu_ram_usage)
        max_gpu_usage.append(max_gpu_ram_usage)
        avg_gpu_usage.append(avg_gpu_ram_usage)

    # Save results
    accuracy_table = pd.DataFrame(accuracy_results)
    print(accuracy_table)
    accuracy_table.to_csv(args.output_csv, index=False)

    # Visualize embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(train_data.to(device))
    numeric_labels_list = [int(value) for value in labels.values()]
    EmbeddingsVisualizer.visualize_embeddings_3d(final_embeddings.cpu().detach().numpy(), numeric_labels_list)

if __name__ == "__main__":
    main()