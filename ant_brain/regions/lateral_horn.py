"""
Lateralhorn (LH) - Angeborene olfaktorische Reaktionen.

Vermittelt hardwired Verhaltensantworten auf Gerüche,
im Gegensatz zum Pilzkörper (gelernte Antworten).

Besonders wichtig für:
- Alarm-Pheromon: Stereotypische Alarmreaktion
- Aggregationspheromon: Anziehung
- Nahrungsgerüche: Appetitive Reaktionen

Die LH erhält direkte PN-Eingabe vom AL und verbindet
zu prämotorischen Bereichen.
"""

import numpy as np
from ant_brain.core.neuron import LIFNeuron, IzhikevichNeuron, NeuronParams
from ant_brain.core.synapse import Synapse


class LateralHorn:
    """
    Lateralhorn - angeborene Geruchsverarbeitung.

    Fest verdrahtete Schaltkreise für stereotypische Reaktionen
    auf biologisch relevante Gerüche.
    """

    # Verhaltenskanäle
    CHANNEL_ALARM = 0       # Flucht/Aggression
    CHANNEL_ATTRACTION = 1  # Annäherung (Nahrung, Trail)
    CHANNEL_AVOIDANCE = 2   # Vermeidung (Gefahr)
    CHANNEL_SOCIAL = 3      # Soziale Interaktion
    N_CHANNELS = 4

    def __init__(self, n_neurons: int = 3000, n_pn_input: int = 2000,
                 pns_per_glom: int = 5, n_glomeruli: int = 400):
        self.n_neurons = n_neurons
        self.n_pn_input = n_pn_input
        self.n_per_channel = n_neurons // self.N_CHANNELS
        self.pns_per_glom = pns_per_glom
        self.n_glomeruli = n_glomeruli
        # l-ALT PNs per glomerulus (last 30% of PNs)
        n_malt_per_glom = int(pns_per_glom * 0.7)
        self.lalt_per_glom = pns_per_glom - n_malt_per_glom

        # Hauptneuronen - lower threshold for easier activation
        self.neurons = LIFNeuron(n_neurons, NeuronParams(
            v_rest=-65.0, v_threshold=-55.0, tau_m=10.0
        ), label="LateralHorn")

        # Ausgabe-Interneuronen (zu prämotorischen Bereichen)
        self.output_neurons = IzhikevichNeuron(self.N_CHANNELS,
            IzhikevichNeuron.REGULAR_SPIKING,
            label="LH_output")

        # PN → LH Synapsen (fest verdrahtet, nicht plastisch) - stronger weights
        # Force dense matrix for innate circuit weight manipulation
        self.syn_pn_lh = Synapse(n_pn_input, n_neurons,
            connectivity=0.2, weight_mean=1.5, excitatory=True,
            label="PN→LH")
        # Ensure dense for _setup_innate_circuits slice indexing
        if self.syn_pn_lh.use_sparse:
            from scipy import sparse
            self.syn_pn_lh.weights = self.syn_pn_lh.weights.toarray()
            self.syn_pn_lh.use_sparse = False

        # LH → Ausgabe Synapsen - stronger to drive output neurons
        self.syn_lh_out = Synapse(n_neurons, self.N_CHANNELS,
            connectivity=0.4, weight_mean=1.5, excitatory=True,
            label="LH→Output")

        # Fest verdrahtete Bias-Gewichte für angeborene Reaktionen
        self._setup_innate_circuits()

        # Ausgabe
        self.channel_activity = np.zeros(self.N_CHANNELS)
        self.dominant_response = -1

    def _setup_innate_circuits(self):
        """Konfiguriere fest verdrahtete Schaltkreise für angeborene Reaktionen.

        PN indices here refer to l-ALT PN positions, which are packed
        sequentially: lalt_per_glom PNs per glomerulus.
        """
        lppg = self.lalt_per_glom  # l-ALT PNs per glomerulus

        # Alarm glomeruli: 0-4 → l-ALT indices [0, 5*lppg)
        alarm_pn_end = min(5 * lppg, self.n_pn_input)
        alarm_pn_range = slice(0, alarm_pn_end)
        alarm_neuron_range = slice(0, self.n_per_channel)
        if alarm_pn_end > 0:
            self.syn_pn_lh.weights[alarm_neuron_range, alarm_pn_range] *= 12.0
            # Suppress alarm PN connections to non-alarm channels
            for ch in range(self.N_CHANNELS):
                if ch != self.CHANNEL_ALARM:
                    ch_range = slice(ch * self.n_per_channel, (ch + 1) * self.n_per_channel)
                    self.syn_pn_lh.weights[ch_range, alarm_pn_range] *= 0.1

        # Trail glomeruli: 5-14 → l-ALT indices [5*lppg, 15*lppg)
        trail_start = 5 * lppg
        trail_end = min(15 * lppg, self.n_pn_input)
        trail_pn_range = slice(trail_start, trail_end)
        attract_range = slice(self.n_per_channel, 2 * self.n_per_channel)
        if trail_end > trail_start:
            self.syn_pn_lh.weights[attract_range, trail_pn_range] *= 8.0
            # Suppress trail connections to non-attraction channels
            for ch in range(self.N_CHANNELS):
                if ch != self.CHANNEL_ATTRACTION:
                    ch_range = slice(ch * self.n_per_channel, (ch + 1) * self.n_per_channel)
                    self.syn_pn_lh.weights[ch_range, trail_pn_range] *= 0.1

        # Food glomeruli: 35-54 → l-ALT indices [35*lppg, 55*lppg)
        food_start = 35 * lppg
        food_end = min(55 * lppg, self.n_pn_input)
        food_pn_range = slice(food_start, food_end)
        if food_end > food_start:
            self.syn_pn_lh.weights[attract_range, food_pn_range] *= 8.0
            # Suppress food connections to non-attraction channels
            for ch in range(self.N_CHANNELS):
                if ch != self.CHANNEL_ATTRACTION:
                    ch_range = slice(ch * self.n_per_channel, (ch + 1) * self.n_per_channel)
                    self.syn_pn_lh.weights[ch_range, food_pn_range] *= 0.1

    def step(self, pn_spikes: np.ndarray, dt: float = 0.1) -> dict:
        """
        Verarbeitung von PN-Eingaben für angeborene Reaktionen.

        Args:
            pn_spikes: Spikes von Projektionsneuronen (l-ALT)

        Returns:
            Dict mit Kanalaktivitäten und dominanter Reaktion
        """
        # PN → LH
        I_lh = self.syn_pn_lh.transmit(pn_spikes)
        lh_spikes = self.neurons.step(I_lh, dt)

        # LH → Ausgabekanäle
        I_out = self.syn_lh_out.transmit(lh_spikes)
        out_spikes = self.output_neurons.step(I_out, dt)

        # Kanalaktivität berechnen (gleitender Mittelwert)
        for ch in range(self.N_CHANNELS):
            ch_slice = slice(ch * self.n_per_channel, (ch + 1) * self.n_per_channel)
            activity = lh_spikes[ch_slice].sum() / self.n_per_channel
            self.channel_activity[ch] = 0.9 * self.channel_activity[ch] + 0.1 * activity

        # Dominante Reaktion bestimmen
        if self.channel_activity.max() > 0.001:
            self.dominant_response = int(np.argmax(self.channel_activity))
        else:
            self.dominant_response = -1

        return {
            'channel_activity': self.channel_activity.copy(),
            'dominant_response': self.dominant_response,
            'response_labels': ['alarm', 'attraction', 'avoidance', 'social'],
            'output_spikes': out_spikes,
        }

    def reset(self):
        self.neurons.reset()
        self.output_neurons.reset()
        self.syn_pn_lh.reset()
        self.syn_lh_out.reset()
        self.channel_activity[:] = 0
        self.dominant_response = -1
