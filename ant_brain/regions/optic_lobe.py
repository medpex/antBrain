"""
Optischer Lobus (Optic Lobe) - Visuelle Verarbeitung.

Drei geschachtelte Neuropile:
- Lamina: Erste Verarbeitungsstufe, retinotop
- Medulla: Farbe, Bewegung, Kantenerkennung
- Lobula: Höhere visuelle Merkmalsextraktion

Ausgabe zu:
- Pilzkörper-Collar (visuelle Lernassoziationen)
- Anteriorer Optischer Tuberkel → Zentralkomplex (Himmelskompass)
"""

import numpy as np
from ant_brain.core.neuron import LIFNeuron, NeuronParams
from ant_brain.core.synapse import Synapse


class OpticLobe:
    """
    Optischer Lobus mit 3-stufiger visueller Verarbeitung.

    Ameisen haben relativ kleine Augen (je nach Art 100-1300 Ommatidien).
    Modell basiert auf einer mittelgroßen Art mit ~600 Ommatidien.
    """

    def __init__(self, n_ommatidia: int = 600,
                 n_lamina_per_omm: int = 6,
                 n_medulla: int = 3000,
                 n_lobula: int = 1500):
        self.n_ommatidia = n_ommatidia
        self.n_lamina = n_ommatidia * n_lamina_per_omm
        self.n_medulla = n_medulla
        self.n_lobula = n_lobula

        # --- Lamina: Erste Verarbeitungsstufe ---
        self.lamina = LIFNeuron(self.n_lamina, NeuronParams(
            v_rest=-65.0, v_threshold=-50.0, tau_m=5.0  # Schnell
        ), label="Lamina")

        # --- Medulla: Zweite Stufe ---
        self.medulla = LIFNeuron(n_medulla, NeuronParams(
            v_rest=-65.0, v_threshold=-50.0, tau_m=10.0
        ), label="Medulla")

        # --- Lobula: Dritte Stufe ---
        self.lobula = LIFNeuron(n_lobula, NeuronParams(
            v_rest=-65.0, v_threshold=-52.0, tau_m=15.0
        ), label="Lobula")

        # Synapsen zwischen Schichten
        self.syn_lam_med = Synapse(self.n_lamina, n_medulla,
            connectivity=0.05, weight_mean=0.6, excitatory=True,
            label="Lamina→Medulla")

        self.syn_med_lob = Synapse(n_medulla, n_lobula,
            connectivity=0.08, weight_mean=0.5, excitatory=True,
            label="Medulla→Lobula")

        # Spezial-Neuronen für Polarisationslicht (DRA - Dorsal Rim Area)
        self.n_dra = 50  # Dorsal Rim Photorezeptoren
        self.dra_neurons = LIFNeuron(self.n_dra, NeuronParams(
            v_rest=-65.0, v_threshold=-48.0, tau_m=8.0
        ), label="DRA_polarization")

        # Ausgabe
        self.visual_output = np.zeros(n_lobula, dtype=bool)
        self.polarization_output = np.zeros(self.n_dra, dtype=bool)

    def step(self, visual_input: np.ndarray,
             polarization_angle: float = 0.0,
             dt: float = 0.1) -> dict:
        """
        Visuelle Verarbeitung.

        Args:
            visual_input: Pixelintensitäten (n_ommatidia,) normalisiert 0-1
            polarization_angle: Polarisationswinkel des Himmels (rad)
            dt: Zeitschritt

        Returns:
            Dict mit visueller Ausgabe und Kompassinfo
        """
        # --- Lamina ---
        # Jedes Ommatidium aktiviert seine Lamina-Neuronen
        I_lamina = np.repeat(visual_input * 5.0, self.n_lamina // self.n_ommatidia)
        if len(I_lamina) < self.n_lamina:
            I_lamina = np.pad(I_lamina, (0, self.n_lamina - len(I_lamina)))
        lamina_spikes = self.lamina.step(I_lamina[:self.n_lamina], dt)

        # --- Medulla ---
        I_medulla = self.syn_lam_med.transmit(lamina_spikes)
        medulla_spikes = self.medulla.step(I_medulla, dt)

        # --- Lobula ---
        I_lobula = self.syn_med_lob.transmit(medulla_spikes)
        lobula_spikes = self.lobula.step(I_lobula, dt)
        self.visual_output = lobula_spikes

        # --- Polarisationslicht (DRA) ---
        # DRA-Neuronen reagieren auf spezifische Polarisationswinkel
        preferred_angles = np.linspace(0, np.pi, self.n_dra)
        pol_response = np.cos(2 * (polarization_angle - preferred_angles))
        pol_response = (pol_response + 1) / 2 * 4.0  # Normalisiert auf 0-4 nA
        pol_spikes = self.dra_neurons.step(pol_response, dt)
        self.polarization_output = pol_spikes

        return {
            'lobula_spikes': lobula_spikes,  # → Pilzkörper Collar
            'polarization_spikes': pol_spikes,  # → Zentralkomplex
            'medulla_activity': medulla_spikes.sum() / self.n_medulla,
            'visual_features': lobula_spikes,
        }

    def reset(self):
        self.lamina.reset()
        self.medulla.reset()
        self.lobula.reset()
        self.dra_neurons.reset()
        self.syn_lam_med.reset()
        self.syn_med_lob.reset()
