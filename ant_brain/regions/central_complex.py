"""
Zentralkomplex (Central Complex) - Navigation und Motorkoordination.

Struktur (4 Neuropile):
- Protocerebralbrücke (PB): 16 Glomeruli, kodiert Kopfrichtung
- Fächerförmiger Körper (FB): Horizontale Schichten × vertikale Säulen
- Ellipsoidkörper (EB): Ringförmig, Head-Direction-Ringattractor
- Gepaarte Noduli: Eigenbewegungssignale (Geschwindigkeit)

Head-Direction-System:
- Ringattractor mit 8 Säulen im EB
- Ein einzelner "Bump" neuronaler Aktivität rotiert mit der Kopfrichtung
- Analog zu Head-Direction-Zellen bei Säugetieren
"""

import numpy as np
from ant_brain.core.neuron import LIFNeuron, IzhikevichNeuron, NeuronParams
from ant_brain.core.synapse import Synapse


class RingAttractor:
    """
    Ring-Attraktor-Schaltung für Kopfrichtungsrepräsentation.

    8 Säulen mit lokaler Erregung und globaler Inhibition erzeugen
    einen stabilen Aktivitätsbump, der die aktuelle Kopfrichtung
    repräsentiert.
    """

    def __init__(self, n_columns: int = 8, neurons_per_column: int = 20):
        self.n_columns = n_columns
        self.n_per_col = neurons_per_column
        self.n_total = n_columns * neurons_per_column

        # Erregungsneuronen (E-PG äquivalent) - lower threshold, short refractory
        self.neurons = LIFNeuron(self.n_total, NeuronParams(
            v_rest=-65.0,
            v_threshold=-55.0,
            tau_m=8.0,
            tau_ref=1.0,
        ), label="EB_columns")

        # Ringförmige Konnektivität
        self._build_ring_connectivity()

        # Aktueller Heading-Winkel (0 - 2π)
        self.heading = 0.0

        # Bump-Aktivität
        self.bump = np.zeros(n_columns)

    def _build_ring_connectivity(self):
        """Erstelle ringförmige exzitatorische und inhibitorische Verbindungen."""
        n = self.n_total
        nc = self.n_columns
        npc = self.n_per_col

        # Exzitatorische Gewichte (lokal, innerhalb und benachbarte Säulen)
        self.w_exc = np.zeros((n, n))
        for i in range(nc):
            for j in range(nc):
                dist = min(abs(i - j), nc - abs(i - j))
                if dist <= 1:
                    # Stärke abhängig von Distanz - stronger for stable bump
                    strength = 1.0 if dist == 0 else 0.5
                    rows = slice(i * npc, (i + 1) * npc)
                    cols = slice(j * npc, (j + 1) * npc)
                    self.w_exc[rows, cols] = strength * (
                        np.random.random((npc, npc)) * 0.3 + 0.7)

        # Globale Inhibition (alle Säulen hemmen alle anderen)
        # Weaker inhibition so the bump can form and persist
        self.w_inh = np.ones((n, n)) * -0.03 / n

    def step(self, compass_input: float, angular_velocity: float,
             dt: float = 0.1) -> dict:
        """
        Ein Zeitschritt des Ringattraktors.

        Args:
            compass_input: Polarisiertes Licht / Kompasswinkel (rad)
            angular_velocity: Drehgeschwindigkeit (rad/s)
            dt: Zeitschritt

        Returns:
            Dict mit Heading-Schätzung und Bump-Aktivität
        """
        # Heading updaten basierend auf Drehung
        self.heading += angular_velocity * dt / 1000.0
        self.heading %= (2 * np.pi)

        # Kompass-Input als sensorischer Drive
        target_column = int((compass_input / (2 * np.pi)) * self.n_columns) % self.n_columns

        # Sensorischer Strom - strong enough to drive spikes quickly
        I_sensory = np.zeros(self.n_total)
        col_start = target_column * self.n_per_col
        I_sensory[col_start:col_start + self.n_per_col] = 8.0

        # Drehungssignal verschiebt den Bump
        shift_signal = angular_velocity * 0.01
        shift_col = int(np.round(shift_signal)) % self.n_columns

        # Rekurrenter Input
        spike_float = self.neurons.spikes.astype(float)
        I_rec = self.w_exc @ spike_float + self.w_inh @ spike_float

        # Gesamtstrom
        I_total = I_sensory + I_rec * 8.0

        # Neuronen updaten
        spikes = self.neurons.step(I_total, dt)

        # Bump-Aktivität pro Säule berechnen (exponential smoothing)
        for i in range(self.n_columns):
            col_slice = slice(i * self.n_per_col, (i + 1) * self.n_per_col)
            instant = spikes[col_slice].sum() / self.n_per_col
            # Smooth: fast rise, slow decay for stable bump readout
            if instant > self.bump[i]:
                self.bump[i] = 0.5 * self.bump[i] + 0.5 * instant
            else:
                self.bump[i] = 0.95 * self.bump[i] + 0.05 * instant

        # Heading aus Bump schätzen
        if self.bump.sum() > 0:
            angles = np.linspace(0, 2 * np.pi, self.n_columns, endpoint=False)
            # Zirkuläres Mittel
            x = np.sum(self.bump * np.cos(angles))
            y = np.sum(self.bump * np.sin(angles))
            estimated_heading = np.arctan2(y, x) % (2 * np.pi)
        else:
            estimated_heading = self.heading

        return {
            'heading': estimated_heading,
            'bump_activity': self.bump.copy(),
            'spikes': spikes,
        }


class CentralComplex:
    """
    Vollständiger Zentralkomplex mit allen 4 Neuropilen.

    Integriert Kompass-, Odometer- und visuelle Eingaben
    für Pfadintegration und Navigationssteuerung.
    """

    def __init__(self, n_columns: int = 16, neurons_per_column: int = 20):
        self.n_columns = n_columns

        # --- Protocerebralbrücke (PB): Heading-Kodierung ---
        self.n_pb = n_columns * 10
        self.pb_neurons = LIFNeuron(self.n_pb, NeuronParams(
            v_rest=-65.0, v_threshold=-50.0, tau_m=12.0
        ), label="PB")

        # --- Ellipsoidkörper (EB): Ring-Attraktor ---
        self.ring_attractor = RingAttractor(
            n_columns=min(n_columns, 8),
            neurons_per_column=neurons_per_column
        )

        # --- Fächerförmiger Körper (FB): Pfadintegration ---
        self.n_fb_layers = 8
        self.n_fb_per_layer = n_columns * 5
        self.n_fb = self.n_fb_layers * self.n_fb_per_layer
        self.fb_neurons = LIFNeuron(self.n_fb, NeuronParams(
            v_rest=-65.0, v_threshold=-50.0, tau_m=15.0
        ), label="FB")

        # --- Noduli: Geschwindigkeitssignal ---
        self.n_noduli = 40  # Bilateral gepaart
        self.noduli = LIFNeuron(self.n_noduli, NeuronParams(
            v_rest=-65.0, v_threshold=-50.0, tau_m=8.0
        ), label="Noduli")

        # --- Pfadintegrationsvektor (Heimvektor) ---
        self.home_vector = np.zeros(2)  # (x, y) Versatz vom Nest
        self.current_speed = 0.0

        # PB → EB Synapsen
        self.syn_pb_eb = Synapse(self.n_pb, self.ring_attractor.n_total,
            connectivity=0.15, weight_mean=0.6, excitatory=True,
            label="PB→EB")

        # EB → FB Synapsen
        self.syn_eb_fb = Synapse(self.ring_attractor.n_total, self.n_fb,
            connectivity=0.1, weight_mean=0.5, excitatory=True,
            label="EB→FB")

        # Steering-Ausgabe (links/rechts)
        self.steering = np.zeros(2)  # [links, rechts]

    def step(self, compass_angle: float, speed: float,
             angular_velocity: float = 0.0, dt: float = 0.1) -> dict:
        """
        Ein Zeitschritt des Zentralkomplexes.

        Args:
            compass_angle: Aktuelle Kompassrichtung (rad)
            speed: Laufgeschwindigkeit
            angular_velocity: Drehgeschwindigkeit (rad/s)
            dt: Zeitschritt

        Returns:
            Dict mit Heading, Heimvektor und Steering
        """
        # --- Ring-Attraktor (Heading) ---
        ra_result = self.ring_attractor.step(compass_angle, angular_velocity, dt)
        heading = ra_result['heading']

        # --- Odometer (Noduli) ---
        self.current_speed = speed
        I_noduli = np.full(self.n_noduli, speed * 2.0)
        noduli_spikes = self.noduli.step(I_noduli, dt)

        # --- Pfadintegration ---
        # Heimvektor aktualisieren
        dx = speed * np.cos(heading) * dt / 1000.0
        dy = speed * np.sin(heading) * dt / 1000.0
        self.home_vector[0] += dx
        self.home_vector[1] += dy

        # --- FB: Vektorkomputation ---
        # Heimvektor-Richtung und -Distanz
        home_dist = np.linalg.norm(self.home_vector)
        if home_dist > 0.01:
            home_angle = np.arctan2(self.home_vector[1], self.home_vector[0])
        else:
            home_angle = 0.0

        # FB erhält EB-Heading + Heimvektor-Info
        I_fb = self.syn_eb_fb.transmit(ra_result['spikes'])
        # Zusätzlicher Input proportional zur Heimvektordistanz
        I_fb += np.random.normal(home_dist * 0.5, 0.1, self.n_fb)
        fb_spikes = self.fb_neurons.step(I_fb, dt)

        # --- Steering-Berechnung ---
        # Differenz zwischen aktueller Richtung und Heimrichtung
        angle_diff = home_angle - heading
        # Normalisierung auf [-π, π]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        self.steering[0] = max(0, -angle_diff)  # Links drehen
        self.steering[1] = max(0, angle_diff)    # Rechts drehen

        return {
            'heading': heading,
            'home_vector': self.home_vector.copy(),
            'home_distance': home_dist,
            'home_angle': home_angle,
            'steering': self.steering.copy(),
            'bump_activity': ra_result['bump_activity'],
            'speed': speed,
        }

    def reset_path_integration(self):
        """Heimvektor zurücksetzen (z.B. wenn Nest erreicht)."""
        self.home_vector[:] = 0

    def reset(self):
        self.ring_attractor.neurons.reset()
        self.pb_neurons.reset()
        self.fb_neurons.reset()
        self.noduli.reset()
        self.home_vector[:] = 0
        self.steering[:] = 0
