import torch

from docless.data.loaders import generate_synthetic
from docless.hypergraph import HGNNEncoder, build_hypergraph
from docless.ssl import CascadeMLM, cascade_ssl_loss, mask_cascade_slices


def test_mask_batch_shapes():
    cascades = generate_synthetic(n_stories=30, seed=2)
    h = build_hypergraph(cascades, feature_dim=16)
    batch = mask_cascade_slices(h, cascades, mask_ratio=0.3, num_negatives=4, seed=2)
    assert batch.anchor_idx.numel() == batch.positive_idx.numel()
    assert batch.negative_idx.shape[1] == 4
    assert batch.anchor_idx.numel() > 0


def test_ssl_loss_decreases_when_overfit_to_one_batch():
    cascades = generate_synthetic(n_stories=40, seed=11)
    h = build_hypergraph(cascades, feature_dim=16)
    batch = mask_cascade_slices(h, cascades, mask_ratio=0.5, num_negatives=4, seed=11)
    assert batch.anchor_idx.numel() > 0

    enc = HGNNEncoder(in_dim=16, hidden_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    head = CascadeMLM(hidden_dim=32)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-2)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        h_nodes = enc(h.x, h.edge_index, h.edge_type, h.node_type)
        loss = cascade_ssl_loss(h_nodes, head, batch)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0] - 0.1, f"SSL loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
