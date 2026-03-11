"""
Antennallobus (AL) - Primäres olfaktorisches Zentrum.

Struktur:
- ~400 Glomeruli organisiert in 7 Clustern
- Olfaktorische Rezeptorneuronen (ORNs) → Glomeruli
- Lokale Interneuronen (LNs): Laterale Inhibition
- Projektionsneuronen (PNs): Ausgabe zu MB und LH

Duale Ausgabewege:
- Medialer ALT (m-ALT): Exzitatorische PNs → Pilzkörper-Calyces + Lateralhorn
- Lateraler ALT (l-ALT): GABAerge PNs → Lateralhorn
"""

import numpy as np
from ant_brain.core.neuron import LIFNeuron, IzhikevichNeuron, NeuronParams
from ant_brain.core.synapse import Synapse


class Glomerulus:
    """Ein einzelner Glomerulus - funktionale Einheit der Geruchsverarbeitung."""

    def __init__(self, glom_id: int, n_orns: int = 120, n_lns: int = 15,
                 n_pns: int = 5):
        self.glom_id = glom_id
        self.n_orns = n_orns
        self.n_lns = n_lns
        self.n_pns = n_pns

        # ORN-Eingangsstrom (von Antenne)
        self.orn_input = np.zeros(n_orns)

        # Lokale Interneuronen (inhibitorisch, GABA)
        self.lns = LIFNeuron(n_lns, NeuronParams(
            v_rest=-65.0, v_threshold=-50.0, tau_m=8.0
        ), label=f"LN_glom{glom_id}")

        # Projektionsneuronen (exzitatorisch, ACh)
        self.pns = IzhikevichNeuron(n_pns,
            IzhikevichNeuron.REGULAR_SPIKING,
            label=f"PN_glom{glom_id}")

        # ORN → PN Synapsen (konvergent) - strong convergent drive
        self.syn_orn_pn = Synapse(n_orns, n_pns,
            connectivity=0.8, weight_mean=2.0, excitatory=True,
            label="ORN→PN")

        # ORN → LN Synapsen
        self.syn_orn_ln = Synapse(n_orns, n_lns,
            connectivity=0.5, weight_mean=1.0, excitatory=True,
            label="ORN→LN")

        # LN → PN Synapsen (inhibitorisch) - moderate inhibition
        self.syn_ln_pn = Synapse(n_lns, n_pns,
            connectivity=0.7, weight_mean=0.3, excitatory=False,
            label="LN→PN")

    def step(self, orn_activation: np.ndarray, lateral_inhibition: float,
             dt: float = 0.1) -> np.ndarray:
        """
        Verarbeitung eines Zeitschritts.

        Args:
            orn_activation: Aktivierung der ORNs dieses Glomerulus
            lateral_inhibition: Inhibitorischer Strom von anderen Glomeruli
            dt: Zeitschritt

        Returns:
            Spike-Ausgabe der Projektionsneuronen
        """
        # ORN → LN und ORN → PN
        # Boost activation to ensure sufficient ORN firing
        boosted_activation = np.clip(orn_activation * 3.0, 0, 1.0)
        orn_spikes = boosted_activation > np.random.random(self.n_orns)

        I_ln = self.syn_orn_ln.transmit(orn_spikes) - lateral_inhibition
        ln_spikes = self.lns.step(I_ln, dt)

        I_pn = (self.syn_orn_pn.transmit(orn_spikes) +
                self.syn_ln_pn.transmit(ln_spikes))
        pn_spikes = self.pns.step(I_pn, dt)

        return pn_spikes


class AntennalLobe:
    """
    Vollständiger Antennallobus mit ~400 Glomeruli.

    Verarbeitet olfaktorische Eingaben von der Antenne und
    sendet über zwei Ausgabewege (m-ALT, l-ALT) an Pilzkörper
    und Lateralhorn.
    """

    # Spezielle Glomerulus-Typen
    ALARM_GLOMERULI = list(range(5))          # 5 Alarm-Pheromon-Glomeruli
    TRAIL_GLOMERULI = list(range(5, 15))      # Trail-Pheromon
    NESTMATE_GLOMERULI = list(range(15, 35))  # Nestkameraden-Erkennung (CHC)
    FOOD_GLOMERULI = list(range(35, 55))      # Nahrungsgerüche
    GENERAL_GLOMERULI = list(range(55, 400))  # Allgemeine Gerüche

    def __init__(self, n_glomeruli: int = 400,
                 orns_per_glom: int = 120,
                 lns_per_glom: int = 15,
                 pns_per_glom: int = 5):
        self.n_glomeruli = n_glomeruli
        self.pns_per_glom = pns_per_glom

        # Alle Glomeruli erstellen
        self.glomeruli = [
            Glomerulus(i, orns_per_glom, lns_per_glom, pns_per_glom)
            for i in range(n_glomeruli)
        ]

        # Laterale Inhibition zwischen Glomeruli (globale LN-Schaltung)
        # Reduced to allow sufficient PN output
        self.global_inhibition_strength = 0.1

        # Ausgabe-Puffer
        self.n_total_pns = n_glomeruli * pns_per_glom
        self.pn_output = np.zeros(self.n_total_pns, dtype=bool)

        # m-ALT vs l-ALT Zuordnung (70% m-ALT exzitatorisch, 30% l-ALT GABAerg)
        self.malt_mask = np.zeros(self.n_total_pns, dtype=bool)
        self.lalt_mask = np.zeros(self.n_total_pns, dtype=bool)
        for i in range(n_glomeruli):
            start = i * pns_per_glom
            n_malt = int(pns_per_glom * 0.7)
            self.malt_mask[start:start + n_malt] = True
            self.lalt_mask[start + n_malt:start + pns_per_glom] = True

    def step(self, odor_vector: np.ndarray, dt: float = 0.1) -> dict:
        """
        Verarbeitung eines Geruchsvektors.

        Args:
            odor_vector: Aktivierungsvektor der Länge n_glomeruli (0-1),
                        repräsentiert die Konzentration verschiedener
                        Duftkomponenten

        Returns:
            Dict mit 'malt_spikes' und 'lalt_spikes' Ausgaben
        """
        # Globale Inhibition berechnen
        total_activity = odor_vector.sum() / self.n_glomeruli
        lateral_inh = total_activity * self.global_inhibition_strength

        # Jeden Glomerulus verarbeiten
        all_pn_spikes = []
        for i, glom in enumerate(self.glomeruli):
            activation = np.full(glom.n_orns, odor_vector[i])
            pn_spikes = glom.step(activation, lateral_inh, dt)
            all_pn_spikes.append(pn_spikes)

        self.pn_output = np.concatenate(all_pn_spikes)

        return {
            'malt_spikes': self.pn_output[self.malt_mask],  # → Pilzkörper + LH
            'lalt_spikes': self.pn_output[self.lalt_mask],  # → Lateralhorn
            'all_pn_spikes': self.pn_output,
        }

    def create_odor_vector(self, odor_type: str, concentration: float = 0.5) -> np.ndarray:
        """Erzeuge einen Geruchsvektor für einen bestimmten Geruchstyp.

        Uses absolute glomerulus ranges that match the class constants
        (ALARM_GLOMERULI, TRAIL_GLOMERULI, etc.) when n_glomeruli >= 400.
        For smaller configs, scales proportionally with at least 1 active glomerulus.
        """
        n = self.n_glomeruli
        vec = np.zeros(n)
        noise = np.random.normal(0, 0.02, n)

        def _range(abs_start, abs_end):
            """Scale absolute indices to current glomerulus count, ensure at least 1."""
            if n >= 400:
                return list(range(abs_start, min(abs_end, n)))
            s = int(abs_start * n / 400)
            e = max(s + 1, int(abs_end * n / 400))
            return list(range(s, min(e, n)))

        if odor_type == 'alarm':
            indices = _range(0, 5)
            vec[indices] = concentration
            vec += np.abs(noise)
        elif odor_type == 'trail':
            indices = _range(5, 15)
            vec[indices] = concentration
            vec += np.abs(noise)
        elif odor_type == 'nestmate':
            indices = _range(15, 35)
            vec[indices] = concentration * 0.5
            vec += np.abs(noise)
        elif odor_type == 'food':
            indices = _range(35, 55)
            vec[indices] = concentration
            vec += np.abs(noise)
        else:
            # Zufälliges Geruchsmuster
            n_active = max(1, int(n * 0.1))
            active = np.random.choice(n, size=n_active, replace=False)
            vec[active] = concentration
            vec += np.abs(noise)

        return np.clip(vec, 0, 1)

    def reset(self):
        for glom in self.glomeruli:
            glom.lns.reset()
            glom.pns.reset()
            glom.syn_orn_pn.reset()
            glom.syn_orn_ln.reset()
            glom.syn_ln_pn.reset()
