"""
Neuronenmodelle für die Ameisengehirn-Simulation.

Implementiert:
- LIF (Leaky Integrate-and-Fire): Für große Populationen (Kenyon-Zellen)
- Izhikevich: Für funktional distinkte Populationen (Projektionsneuronen, MBONs)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NeuronParams:
    """Parameter für Neuronenmodelle."""
    v_rest: float = -65.0       # Ruhepotential (mV)
    v_threshold: float = -50.0  # Schwellenpotential (mV)
    v_reset: float = -65.0      # Reset-Potential nach Spike (mV)
    tau_m: float = 10.0         # Membranzeitkonstante (ms)
    tau_ref: float = 2.0        # Refraktärzeit (ms)
    r_m: float = 10.0           # Membranwiderstand (MOhm)


class LIFNeuron:
    """
    Leaky Integrate-and-Fire Neuron.

    Verwendet für große Neuronenpopulationen wie Kenyon-Zellen
    im Pilzkörper (~60.000-100.000 Neuronen).

    dV/dt = -(V - V_rest) / tau_m + R_m * I / tau_m
    """

    def __init__(self, n_neurons: int, params: Optional[NeuronParams] = None,
                 label: str = ""):
        self.n_neurons = n_neurons
        self.params = params or NeuronParams()
        self.label = label

        # Zustandsvariablen
        self.v = np.full(n_neurons, self.params.v_rest)  # Membranpotential
        self.spikes = np.zeros(n_neurons, dtype=bool)     # Spike-Ausgabe
        self.refractory = np.zeros(n_neurons)              # Refraktärzähler

        # Neuromodulation (DA, OA, 5-HT, TA)
        self.modulation = {
            'dopamine': 1.0,
            'octopamine': 1.0,
            'serotonin': 1.0,
            'tyramine': 1.0,
        }

        # Spike-Historie für Analyse
        self.spike_history: list[np.ndarray] = []
        self.spike_count = np.zeros(n_neurons, dtype=int)

    def step(self, I_ext: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Ein Simulationsschritt.

        Args:
            I_ext: Externer Strom für jedes Neuron (nA)
            dt: Zeitschritt (ms)

        Returns:
            Boolean-Array der Spikes
        """
        p = self.params

        # Neuromodulationseffekte auf Erregbarkeit
        # OA erhöht Erregbarkeit (cAMP), DA wirkt primär auf Synapsen (hier nur leichter Gain)
        gain = 0.7 + self.modulation['octopamine'] * 0.2 + self.modulation['dopamine'] * 0.1
        I_mod = I_ext * gain

        # Refraktäre Neuronen nicht updaten
        active = self.refractory <= 0

        # Membranpotential-Update (Euler-Integration)
        dv = (-(self.v - p.v_rest) + p.r_m * I_mod) / p.tau_m * dt
        self.v[active] += dv[active]

        # Spike-Detektion
        self.spikes = self.v >= p.v_threshold

        # Reset nach Spike
        self.v[self.spikes] = p.v_reset
        self.refractory[self.spikes] = p.tau_ref

        # Refraktärzeit dekrementieren
        self.refractory -= dt
        self.refractory = np.maximum(self.refractory, 0)

        # Spike-Zähler und Historie (capped to prevent memory growth)
        self.spike_count += self.spikes.astype(int)
        self.spike_history.append(self.spikes.copy())
        if len(self.spike_history) > 1000:
            self.spike_history = self.spike_history[-500:]

        return self.spikes

    def reset(self):
        """Zustand zurücksetzen."""
        self.v[:] = self.params.v_rest
        self.spikes[:] = False
        self.refractory[:] = 0
        self.spike_history.clear()
        self.spike_count[:] = 0

    def set_modulation(self, neurotransmitter: str, level: float):
        """Neuromodulationslevel setzen (0.0 - 2.0)."""
        if neurotransmitter in self.modulation:
            self.modulation[neurotransmitter] = np.clip(level, 0.0, 2.0)

    def get_firing_rate(self, window_ms: float = 100.0, dt: float = 0.1) -> np.ndarray:
        """Mittlere Feuerrate über ein Zeitfenster."""
        steps = int(window_ms / dt)
        if len(self.spike_history) < steps:
            steps = len(self.spike_history)
        if steps == 0:
            return np.zeros(self.n_neurons)
        recent = np.array(self.spike_history[-steps:])
        return recent.sum(axis=0) / (window_ms / 1000.0)


class IzhikevichNeuron:
    """
    Izhikevich-Neuronenmodell.

    Reichere Dynamik (Bursting, Chattering) für funktional
    distinkte Populationen wie Projektionsneuronen und MBONs.

    dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    du/dt = a*(b*v - u)
    if v >= 30: v = c, u = u + d
    """

    # Vordefinierte Neuronentypen
    REGULAR_SPIKING = {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0}
    FAST_SPIKING = {'a': 0.1, 'b': 0.2, 'c': -65.0, 'd': 2.0}
    BURSTING = {'a': 0.02, 'b': 0.2, 'c': -55.0, 'd': 4.0}
    CHATTERING = {'a': 0.02, 'b': 0.2, 'c': -50.0, 'd': 2.0}

    def __init__(self, n_neurons: int, neuron_type: Optional[dict] = None,
                 label: str = ""):
        self.n_neurons = n_neurons
        self.label = label

        nt = neuron_type or self.REGULAR_SPIKING
        self.a = np.full(n_neurons, nt['a'])
        self.b = np.full(n_neurons, nt['b'])
        self.c = np.full(n_neurons, nt['c'])
        self.d = np.full(n_neurons, nt['d'])

        # Zustandsvariablen
        self.v = np.full(n_neurons, -65.0)
        self.u = self.b * self.v
        self.spikes = np.zeros(n_neurons, dtype=bool)

        self.modulation = {
            'dopamine': 1.0,
            'octopamine': 1.0,
            'serotonin': 1.0,
        }

        self.spike_history: list[np.ndarray] = []
        self.spike_count = np.zeros(n_neurons, dtype=int)

    def step(self, I_ext: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Ein Simulationsschritt (0.5ms Substeps für Stabilität)."""
        gain = self.modulation['octopamine'] * self.modulation['dopamine']
        I_mod = I_ext * gain

        # Halber Schritt für numerische Stabilität
        half_dt = dt / 2.0
        for _ in range(2):
            dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I_mod)
            self.v += dv * half_dt
            du = self.a * (self.b * self.v - self.u)
            self.u += du * half_dt

        # Spike-Detektion
        self.spikes = self.v >= 30.0
        self.v[self.spikes] = self.c[self.spikes]
        self.u[self.spikes] += self.d[self.spikes]

        self.spike_count += self.spikes.astype(int)
        self.spike_history.append(self.spikes.copy())
        if len(self.spike_history) > 1000:
            self.spike_history = self.spike_history[-500:]

        return self.spikes

    def reset(self):
        """Zustand zurücksetzen."""
        self.v[:] = -65.0
        self.u = self.b * self.v
        self.spikes[:] = False
        self.spike_history.clear()
        self.spike_count[:] = 0

    def set_modulation(self, neurotransmitter: str, level: float):
        if neurotransmitter in self.modulation:
            self.modulation[neurotransmitter] = np.clip(level, 0.0, 2.0)
