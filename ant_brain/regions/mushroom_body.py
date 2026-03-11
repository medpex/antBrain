"""
Pilzkörper (Mushroom Body) - Lernen, Gedächtnis, sensorische Integration.

Die größte Hirnregion der Ameise (~24-36% aller Neuronen).

Struktur:
- Calyces (Lip, Collar, Basal Ring): Eingaberegionen
  - Lip: Olfaktorischer Input von AL
  - Collar: Visueller Input von OL
  - Basal Ring: Multimodaler Input
- Kenyon-Zellen (KCs): Intrinsische Neuronen, sparse Kodierung
- Mushroom Body Output Neurons (MBONs): ~21 Typen
- Dopaminerge Neuronen (DANs): ~20 Typen, Belohnung/Bestrafung

Lernprinzip:
- Sparse, hochdimensionale Kodierung in KCs (~5-10% Aktivierung)
- Jede KC erhält Input von ~7-10 PNs (divergent, zufällig)
- STDP an KC→MBON Synapsen (dopaminmoduliert)
"""

import numpy as np
from ant_brain.core.neuron import LIFNeuron, IzhikevichNeuron, NeuronParams
from ant_brain.core.synapse import Synapse, STDPSynapse


class MushroomBody:
    """
    Pilzkörper - assoziatives Lernzentrum.

    Implementiert sparse Kodierung und dopaminmoduliertes STDP
    für One-Shot-Lernen.
    """

    def __init__(self, n_kenyon_cells: int = 80000,
                 n_pn_input: int = 2000,
                 n_visual_input: int = 500,
                 n_mbon_types: int = 21,
                 n_dan_types: int = 20,
                 sparseness: float = 0.05):
        """
        Args:
            n_kenyon_cells: Anzahl Kenyon-Zellen
            n_pn_input: Anzahl eingehender Projektionsneuronen (von AL)
            n_visual_input: Anzahl visueller Eingabeneuronen (von OL)
            n_mbon_types: Anzahl MBON-Typen
            n_dan_types: Anzahl DAN-Typen
            sparseness: Ziel-Aktivierungsdichte der KCs (5-10%)
        """
        self.n_kc = n_kenyon_cells
        self.n_pn = n_pn_input
        self.n_visual = n_visual_input
        self.n_mbon = n_mbon_types
        self.n_dan = n_dan_types
        self.sparseness = sparseness

        # --- Kenyon-Zellen (LIF, große Population) ---
        self.kenyon_cells = LIFNeuron(n_kenyon_cells, NeuronParams(
            v_rest=-65.0,
            v_threshold=-55.0,  # Lower threshold to allow sparse but real firing
            tau_m=12.0,
            tau_ref=3.0,
        ), label="KenyonCells")

        # --- MBONs (Izhikevich, kleine Population) ---
        self.mbons = IzhikevichNeuron(n_mbon_types,
            IzhikevichNeuron.REGULAR_SPIKING,
            label="MBONs")

        # --- DANs (Izhikevich) ---
        self.dans = IzhikevichNeuron(n_dan_types,
            IzhikevichNeuron.BURSTING,
            label="DANs")

        # --- Eingabesynapsen (Calyx) ---

        # PN → KC (divergent, sparse, zufällig: jede KC von ~7-10 PNs)
        # Higher connectivity so each KC receives enough PN input
        pn_kc_connectivity = min(40.0 / n_pn_input, 1.0)
        self.syn_pn_kc = Synapse(n_pn_input, n_kenyon_cells,
            connectivity=pn_kc_connectivity,
            weight_mean=2.5,
            weight_std=0.5,
            excitatory=True,
            label="PN→KC (Lip)")

        # Visueller Input → KC (Collar)
        vis_kc_connectivity = min(5.0 / n_visual_input, 1.0)
        self.syn_vis_kc = Synapse(n_visual_input, n_kenyon_cells,
            connectivity=vis_kc_connectivity,
            weight_mean=0.5,
            weight_std=0.15,
            excitatory=True,
            label="Visual→KC (Collar)")

        # --- Ausgabesynapsen (Lobes) ---

        # KC → MBON (STDP, dopaminmoduliert - DAS zentrale Lernsynapse)
        self.syn_kc_mbon = STDPSynapse(n_kenyon_cells, n_mbon_types,
            connectivity=0.3,
            weight_mean=0.8,
            weight_std=0.2,
            excitatory=True,
            tau_plus=20.0,
            tau_minus=25.0,
            a_plus=0.05,
            a_minus=0.06,
            w_max=2.0,
            label="KC→MBON (STDP)")

        # DAN → KC-MBON Modulation (konzeptuell)
        self.dan_targets = np.random.randint(0, n_mbon_types, n_dan_types)

        # --- Inhibitorisches Feedback für Sparseness ---
        # Globale inhibitorische Rückkopplung (APL-ähnlich wie in Drosophila)
        # Reduced to allow initial firing before feedback kicks in
        self.global_inhibition = 0.0
        self.inhibition_strength = 0.15

        # Ausgabe
        self.mbon_output = np.zeros(n_mbon_types)

    def step(self, pn_spikes: np.ndarray,
             visual_spikes: np.ndarray = None,
             reward_signal: float = 0.0,
             dt: float = 0.1) -> dict:
        """
        Verarbeitung eines Zeitschritts.

        Args:
            pn_spikes: Spikes von Projektionsneuronen (AL-Ausgabe)
            visual_spikes: Spikes von visuellen Neuronen (OL-Ausgabe)
            reward_signal: Belohnungssignal (-1 bis +1)
            dt: Zeitschritt

        Returns:
            Dict mit MBON-Ausgabe und Lernmetriken
        """
        # --- Calyx: Input-Integration ---
        I_kc = self.syn_pn_kc.transmit(pn_spikes)

        if visual_spikes is not None:
            I_kc += self.syn_vis_kc.transmit(visual_spikes)

        # Globale Inhibition für sparse Kodierung
        I_kc -= self.global_inhibition * self.inhibition_strength

        # KC-Aktivierung
        kc_spikes = self.kenyon_cells.step(I_kc, dt)

        # Globale Inhibition updaten (APL-like feedback)
        kc_activity = kc_spikes.sum() / self.n_kc
        excess = max(0, kc_activity - self.sparseness)
        target_inh = excess * 2.0
        # Fast rise when over target, fast decay when under
        alpha = 0.3 if target_inh > self.global_inhibition else 0.5
        self.global_inhibition += alpha * (target_inh - self.global_inhibition)
        # Cap inhibition to prevent permanent suppression
        self.global_inhibition = min(self.global_inhibition, 1.0)

        # --- Lobes: KC → MBON ---
        I_mbon = self.syn_kc_mbon.transmit(kc_spikes)
        mbon_spikes = self.mbons.step(I_mbon, dt)

        # --- Dopamin-System ---
        # Belohnungssignal aktiviert DANs (kompartiment-spezifisch)
        if abs(reward_signal) > 0.01:
            # Positive Belohnung → appetitive DANs (PAM-äquivalent)
            # Negative Belohnung → aversive DANs (PPL1-äquivalent)
            I_dan = np.zeros(self.n_dan)
            if reward_signal > 0:
                I_dan[:self.n_dan//2] = reward_signal * 5.0  # PAM-Cluster
            else:
                I_dan[self.n_dan//2:] = abs(reward_signal) * 5.0  # PPL1-Cluster
            dan_spikes = self.dans.step(I_dan, dt)

            # DA-Level basierend auf aktiven DANs und ihren Ziel-MBONs
            da_level = 1.0 + reward_signal
            self.syn_kc_mbon.set_dopamine(da_level)
        else:
            I_dan = np.zeros(self.n_dan)
            self.dans.step(I_dan, dt)
            self.syn_kc_mbon.set_dopamine(0.5)  # Baseline

        # --- STDP-Update ---
        self.syn_kc_mbon.update_stdp(kc_spikes, mbon_spikes, dt)

        # Metriken
        sparseness_actual = kc_spikes.sum() / self.n_kc

        # Gleitender Mittelwert statt binärer Spike-Ausgabe
        self.mbon_output = 0.9 * self.mbon_output + 0.1 * mbon_spikes.astype(float)

        return {
            'mbon_spikes': mbon_spikes,
            'mbon_values': self.mbon_output,
            'kc_sparseness': sparseness_actual,
            'kc_active': int(kc_spikes.sum()),
            'kc_spikes': kc_spikes,
        }

    def get_learned_value(self) -> np.ndarray:
        """Aktuelle gelernte Bewertung jedes MBON-Kanals."""
        return self.mbon_output

    def reset(self):
        self.kenyon_cells.reset()
        self.mbons.reset()
        self.dans.reset()
        self.syn_pn_kc.reset()
        self.syn_vis_kc.reset()
        self.syn_kc_mbon.reset()
        self.global_inhibition = 0.0
