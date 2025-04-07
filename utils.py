import os.path as osp
import numpy as np
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import plotly.graph_objects as go

class GraphLoader:
    """
    Class to load graph data from .npz file.

    Args:
        filepath (str): Path to the .npz file.
    """
    def __init__(self, filepath: str):
        self.filepath = self._resolve_path(filepath)
        self.graph = None
        self.node_labels = {}

    def _resolve_path(self, filepath: str) -> str:
        filepath = osp.abspath(osp.expanduser(filepath))
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        if not osp.isfile(filepath):
            raise ValueError(f"File not found: {filepath}")
        return filepath

    def load_graph(self) -> tuple[nx.Graph, dict[int, str]]:
        try:
            with np.load(self.filepath, allow_pickle=True) as loader:
                loader = dict(loader)
                adj_matrix = csr_matrix(
                    (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                    shape=loader['adj_shape']
                )
                self.graph = nx.from_scipy_sparse_array(adj_matrix)
                if 'labels' in loader:
                    labels = loader['labels']
                    self.node_labels = {i: str(labels[i]) for i in range(len(labels))}
                else:
                    self.node_labels = {}
        except FileNotFoundError:
            raise ValueError(f"File not found: {self.filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading graph: {e}")
        return self.graph, self.node_labels

    def to_pyg_data(self) -> Data:
        if self.graph is None:
            raise RuntimeError("Graph not loaded. Call `load_graph()` first.")
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(list(self.node_labels.values()))
        y = torch.tensor(encoded_labels, dtype=torch.long)
        edge_index = torch.tensor(list(self.graph.edges), dtype=torch.long).t().contiguous()
        num_nodes = self.graph.number_of_nodes()
        x = torch.eye(num_nodes)
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

class EmbeddingEvaluator:
    """
    Class to evaluate embeddings using various metrics.

    Args:
        embeddings (torch.Tensor): Embeddings tensor.
        labels (torch.Tensor): Labels tensor.
    """
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.X = embeddings.detach().cpu().numpy()
        self.y = labels.detach().cpu().numpy()

    def evaluate_accuracy(self, test_size=0.2, random_state=42) -> float:
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        clf = LogisticRegression(multi_class='ovr', solver='liblinear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        return acc

    def evaluate_f1_kfold(self, n_splits=3, random_state=42) -> float:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        f1_scores = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(self.X)):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            clf = LogisticRegression(multi_class='ovr', solver='liblinear')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            f1_scores.append(f1)
            print(f"Fold {fold + 1} F1-Score: {f1:.4f}")
        avg_f1 = np.mean(f1_scores)
        print(f"\nAverage F1-Score across {n_splits} folds: {avg_f1:.4f}")
        return avg_f1

class EmbeddingsVisualizer:
    """
    Class to visualize embeddings in 3D using t-SNE.
    """
    @staticmethod
    def visualize_embeddings_3d(embeddings, labels):
        tsne = TSNE(n_components=3, random_state=42, perplexity=50)
        tsne_embeddings = tsne.fit_transform(embeddings)
        x, y, z = tsne_embeddings[:, 0], tsne_embeddings[:, 1], tsne_embeddings[:,2]
        scatter_trace = go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(size=5, color=labels, colorscale='Viridis', opacity=0.8)
        )
        layout = go.Layout(
            title='Embeddings Visualization (3D)',
            scene=dict(xaxis=dict(title='t-SNE Dimension 1'),
                       yaxis=dict(title='t-SNE Dimension 2'),
                       zaxis=dict(title='t-SNE Dimension 3'))
        )
        fig = go.Figure(data=[scatter_trace], layout=layout)
        fig.show()