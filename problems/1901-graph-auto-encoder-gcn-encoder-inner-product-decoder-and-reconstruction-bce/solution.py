# Cell 1
import numpy as np


def gae_encode(A, X, W1, W2):
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    W2 = np.asarray(W2, dtype=float)

    n = A.shape[0]

    # Add self-loops: A_tilde = A + I
    a_tilde = A + np.eye(n)

    # Degree matrix of A_tilde
    degrees = a_tilde.sum(axis=1)

    # D_tilde^(-1/2)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

    # Symmetric normalization
    a_norm = d_inv_sqrt @ a_tilde @ d_inv_sqrt

    # Two-layer GCN encoder
    hidden = a_norm @ X @ W1
    hidden = np.maximum(hidden, 0.0)  # ReLU
    z = a_norm @ hidden @ W2

    return z


def gae_logits(Z, pairs):
    Z = np.asarray(Z, dtype=float)
    pairs = np.asarray(pairs, dtype=int)

    return np.array([Z[i] @ Z[j] for i, j in pairs], dtype=float)


def _bce_with_logits(logits, targets, pos_weight=1.0):
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=float)

    # Stable log(sigmoid(logits))
    log_sigmoid = -np.logaddexp(0.0, -logits)

    # Stable log(1 - sigmoid(logits)) = log(sigmoid(-logits))
    log_sigmoid_neg = -np.logaddexp(0.0, logits)

    loss = -(
        pos_weight * targets * log_sigmoid
        + (1.0 - targets) * log_sigmoid_neg
    )

    return loss


def gae_loss(A, Z, pos_weight=None):
    A = np.asarray(A, dtype=float)
    Z = np.asarray(Z, dtype=float)

    n = A.shape[0]
    logits = Z @ Z.T

    if pos_weight is None:
        pos_weight = (n**2 - A.sum()) / A.sum()

    losses = _bce_with_logits(logits, A, pos_weight=pos_weight)

    return float(losses.mean())


def sampled_gae_loss(Z, pos_pairs, neg_pairs):
    Z = np.asarray(Z, dtype=float)

    pos_logits = gae_logits(Z, pos_pairs)
    neg_logits = gae_logits(Z, neg_pairs)

    pos_targets = np.ones(len(pos_pairs), dtype=float)
    neg_targets = np.zeros(len(neg_pairs), dtype=float)

    pos_losses = _bce_with_logits(pos_logits, pos_targets, pos_weight=1.0)
    neg_losses = _bce_with_logits(neg_logits, neg_targets, pos_weight=1.0)

    all_losses = np.concatenate([pos_losses, neg_losses])

    return float(all_losses.mean())


def vgae_sample(mu, logvar, eps):
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    eps = np.asarray(eps, dtype=float)

    return mu + np.exp(0.5 * logvar) * eps


def vgae_kl(mu, logvar):
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)

    # Sum KL over latent dimensions, then average over nodes
    kl_per_node = -0.5 * np.sum(
        1.0 + logvar - mu**2 - np.exp(logvar),
        axis=1
    )

    return float(np.mean(kl_per_node))