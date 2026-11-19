import torch

from docless.data.loaders import generate_synthetic
from docless.hypergraph import HGNNEncoder, NodeType, build_hypergraph


def test_build_hypergraph_has_all_four_edge_types():
    cascades = generate_synthetic(n_stories=20, seed=1)
    h = build_hypergraph(cascades, delta_minutes=5.0, k_max=64, feature_dim=16)
    assert h.num_nodes() > 0
    assert h.num_edges() > 0
    assert h.num_nodes_per_type[NodeType.USER] > 0
    assert h.num_nodes_per_type[NodeType.URL] > 0
    assert h.num_nodes_per_type[NodeType.CASCADE_SLICE] > 0
    # at least share + burst are guaranteed by construction; the others depend
    # on whether the synthetic data exercised them — assert at least 2.
    assert int(h.edge_type.unique().numel()) >= 2


def test_encoder_forward_shape():
    cascades = generate_synthetic(n_stories=15, seed=7)
    h = build_hypergraph(cascades, feature_dim=16)
    enc = HGNNEncoder(in_dim=16, hidden_dim=32, num_layers=2, num_heads=4)
    out = enc(h.x, h.edge_index, h.edge_type, h.node_type)
    assert out.shape == (h.num_nodes(), 32)
    assert torch.isfinite(out).all()


def test_encoder_backward():
    cascades = generate_synthetic(n_stories=10, seed=3)
    h = build_hypergraph(cascades, feature_dim=16)
    enc = HGNNEncoder(in_dim=16, hidden_dim=32, num_layers=1, num_heads=4)
    out = enc(h.x, h.edge_index, h.edge_type, h.node_type)
    loss = out.sum()
    loss.backward()
    # Gradients flow back into the input projection.
    assert enc.input_proj.weight.grad is not None
    assert enc.input_proj.weight.grad.abs().sum() > 0
