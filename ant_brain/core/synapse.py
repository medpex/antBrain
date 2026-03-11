"""
Synaptische Verbindungen und Lernregeln.

Implementiert:
- Statische Synapsen (sparse für große Netzwerke)
- STDP (Spike-Timing-Dependent Plasticity) für Lernen im Pilzkörper
"""

import numpy as np
from scipy import sparse
from typing import Optional


def _estimate_elements(n_post, n_pre, connectivity):
    return int(n_post * n_pre * connectivity * 1.1) + 100


class Synapse:
    """
    Statische synaptische Verbindung zwischen Neuronenpopulationen.
    Verwendet scipy.sparse für große Matrizen, dense für kleine.
    """

    SPARSE_THRESHOLD = 500_000  # ab 500k Einträgen sparse verwenden

    def __init__(self, n_pre: int, n_post: int,
                 connectivity: float = 0.1,
                 weight_mean: float = 0.5,
                 weight_std: float = 0.1,
                 excitatory: bool = True,
                 delay_ms: float = 1.0,
                 label: str = ""):
        self.n_pre = n_pre
        self.n_post = n_post
        self.excitatory = excitatory
        self.delay_ms = delay_ms
        self.label = label
        self.use_sparse = (n_post * n_pre) > self.SPARSE_THRESHOLD

        if self.use_sparse:
            self._init_sparse(connectivity, weight_mean, weight_std)
        else:
            self._init_dense(connectivity, weight_mean, weight_std)

        # Verzögerungspuffer
        self.delay_steps = max(1, int(delay_ms / 0.5))
        self.buffer = np.zeros((self.delay_steps, n_pre), dtype=bool)
        self.buffer_idx = 0

    def _init_dense(self, connectivity, weight_mean, weight_std):
        mask = np.random.random((self.n_post, self.n_pre)) < connectivity
        weights = np.abs(np.random.normal(weight_mean, weight_std,
                                          (self.n_post, self.n_pre)))
        if not self.excitatory:
            weights = -weights
        self.weights = weights * mask

    def _init_sparse(self, connectivity, weight_mean, weight_std):
        nnz = _estimate_elements(self.n_post, self.n_pre, connectivity)
        rows = np.random.randint(0, self.n_post, nnz)
        cols = np.random.randint(0, self.n_pre, nnz)
        vals = np.abs(np.random.normal(weight_mean, weight_std, nnz))
        if not self.excitatory:
            vals = -vals
        self.weights = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(self.n_post, self.n_pre))

    def transmit(self, pre_spikes: np.ndarray) -> np.ndarray:
        delayed_spikes = self.buffer[self.buffer_idx]
        self.buffer[self.buffer_idx] = pre_spikes
        self.buffer_idx = (self.buffer_idx + 1) % self.delay_steps

        spike_float = delayed_spikes.astype(np.float64)
        if self.use_sparse:
            return np.asarray(self.weights @ spike_float).ravel()
        else:
            return self.weights @ spike_float

    def reset(self):
        self.buffer[:] = False
        self.buffer_idx = 0


class STDPSynapse:
    """
    Synapse mit Spike-Timing-Dependent Plasticity.
    Optimierte Implementierung für KC→MBON (viele pre, wenige post).

    Dopamin-moduliertes STDP (Three-Factor Learning Rule):
    - Pre vor Post: LTP, moduliert durch DA
    - Post vor Pre: LTD
    """

    def __init__(self, n_pre: int, n_post: int,
                 connectivity: float = 0.1,
                 weight_mean: float = 0.5,
                 weight_std: float = 0.1,
                 excitatory: bool = True,
                 tau_plus: float = 20.0,
                 tau_minus: float = 20.0,
                 a_plus: float = 0.01,
                 a_minus: float = 0.012,
                 w_max: float = 1.0,
                 w_min: float = 0.0,
                 delay_ms: float = 1.0,
                 label: str = ""):
        self.n_pre = n_pre
        self.n_post = n_post
        self.excitatory = excitatory
        self.label = label
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = w_max
        self.w_min = w_min

        # Gewichte: für jedes post-Neuron ein sparse Vektor von pre-Gewichten
        self.conn_indices = []  # pro post: array von pre-Indizes
        self.conn_weights = []  # pro post: array von Gewichten
        for j in range(n_post):
            mask = np.random.random(n_pre) < connectivity
            indices = np.where(mask)[0]
            w = np.abs(np.random.normal(weight_mean, weight_std, len(indices)))
            if not excitatory:
                w = -w
            self.conn_indices.append(indices)
            self.conn_weights.append(w)

        # Verzögerungspuffer
        self.delay_steps = max(1, int(delay_ms / 0.5))
        self.buffer = np.zeros((self.delay_steps, n_pre), dtype=bool)
        self.buffer_idx = 0

        # STDP-Traces
        self.trace_pre = np.zeros(n_pre)
        self.trace_post = np.zeros(n_post)

        # Dopamin-Signal
        self.dopamine_level = 1.0

    def transmit(self, pre_spikes: np.ndarray) -> np.ndarray:
        delayed = self.buffer[self.buffer_idx]
        self.buffer[self.buffer_idx] = pre_spikes
        self.buffer_idx = (self.buffer_idx + 1) % self.delay_steps

        result = np.zeros(self.n_post)
        spike_float = delayed.astype(np.float64)
        for j in range(self.n_post):
            idx = self.conn_indices[j]
            if len(idx) > 0:
                result[j] = self.conn_weights[j] @ spike_float[idx]
        return result

    def update_stdp(self, pre_spikes: np.ndarray, post_spikes: np.ndarray,
                    dt: float = 0.5):
        # Traces decay
        decay_pre = np.exp(-dt / self.tau_plus)
        decay_post = np.exp(-dt / self.tau_minus)
        self.trace_pre *= decay_pre
        self.trace_post *= decay_post

        self.trace_pre[pre_spikes] += 1.0
        self.trace_post[post_spikes] += 1.0

        # Gewichtsupdate nur für aktive post-Neuronen
        post_active = np.where(post_spikes)[0]
        pre_active = np.where(pre_spikes)[0]

        da = self.dopamine_level

        # LTP: post spikt → pre-trace verstärkt Verbindung
        for j in post_active:
            idx = self.conn_indices[j]
            if len(idx) > 0:
                self.conn_weights[j] += self.a_plus * da * self.trace_pre[idx] * dt

        # LTD: pre spikt → post-trace schwächt Verbindung
        if len(pre_active) > 0:
            pre_set = set(pre_active)
            for j in range(self.n_post):
                idx = self.conn_indices[j]
                if len(idx) > 0:
                    # Finde Überlappung
                    in_pre = np.isin(idx, pre_active)
                    if np.any(in_pre):
                        self.conn_weights[j][in_pre] -= (
                            self.a_minus * da * self.trace_post[j] * dt)

        # Gewichte clippen
        for j in range(self.n_post):
            if self.excitatory:
                self.conn_weights[j] = np.clip(
                    self.conn_weights[j], self.w_min, self.w_max)
            else:
                self.conn_weights[j] = np.clip(
                    self.conn_weights[j], -self.w_max, -self.w_min)

    def set_dopamine(self, level: float):
        self.dopamine_level = level

    def get_mean_weight(self) -> float:
        total = sum(w.sum() for w in self.conn_weights)
        count = sum(len(w) for w in self.conn_weights)
        return total / max(count, 1)

    def reset(self):
        self.buffer[:] = False
        self.buffer_idx = 0
        self.trace_pre[:] = 0
        self.trace_post[:] = 0
