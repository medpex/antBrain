"""
Subesophagealganglion (SEG/GNG) - Motorsteuerung der Mundwerkzeuge.

Kontrolliert:
- Mandibeln: Schnelles Zubeißen (Verteidigung) + präzises Greifen (Tragen)
- Maxillen und Labium: Fütterung
- Pharynxpumpe: Nahrungsaufnahme
- Gustatorische Verarbeitung (Geschmack)

Motorische Organisation:
- 10-12 Motorneuronen pro Mandibel-Schließmuskel
- 4-5 schnelle Motorneuronen (rapid snapping)
- Langsame Motorneuronen (präzises Greifen)
"""

import numpy as np
from ant_brain.core.neuron import LIFNeuron, IzhikevichNeuron, NeuronParams
from ant_brain.core.synapse import Synapse


class SubesophagealGanglion:
    """
    Subesophagealganglion - Mundwerkzeugsteuerung und gustatorische Verarbeitung.
    """

    def __init__(self, n_motor: int = 24, n_gustatory: int = 200,
                 n_interneurons: int = 500):
        self.n_motor = n_motor
        self.n_gustatory = n_gustatory
        self.n_interneurons = n_interneurons

        # --- Motorneuronen ---
        # Schnelle Mandibel-Motorneuronen (Zubeißen)
        self.n_fast_motor = 10
        self.fast_motor = LIFNeuron(self.n_fast_motor, NeuronParams(
            v_rest=-65.0, v_threshold=-45.0, tau_m=5.0, tau_ref=1.0
        ), label="FastMotor_Mandible")

        # Langsame Mandibel-Motorneuronen (Greifen)
        self.n_slow_motor = 4
        self.slow_motor = LIFNeuron(self.n_slow_motor, NeuronParams(
            v_rest=-65.0, v_threshold=-50.0, tau_m=15.0, tau_ref=3.0
        ), label="SlowMotor_Mandible")

        # Fütterungs-Motorneuronen
        self.n_feeding_motor = n_motor - self.n_fast_motor - self.n_slow_motor
        self.feeding_motor = LIFNeuron(self.n_feeding_motor, NeuronParams(
            v_rest=-65.0, v_threshold=-50.0, tau_m=10.0
        ), label="FeedingMotor")

        # --- Gustatorische Interneuronen ---
        self.gustatory = LIFNeuron(n_gustatory, NeuronParams(
            v_rest=-65.0, v_threshold=-50.0, tau_m=10.0
        ), label="Gustatory")

        # --- Interneuronen ---
        self.interneurons = LIFNeuron(n_interneurons, NeuronParams(
            v_rest=-65.0, v_threshold=-50.0, tau_m=10.0
        ), label="SEG_interneurons")

        # Synapsen
        self.syn_gust_inter = Synapse(n_gustatory, n_interneurons,
            connectivity=0.1, weight_mean=0.5, excitatory=True,
            label="Gust→Inter")

        self.syn_inter_fast = Synapse(n_interneurons, self.n_fast_motor,
            connectivity=0.2, weight_mean=0.6, excitatory=True,
            label="Inter→FastMotor")

        self.syn_inter_slow = Synapse(n_interneurons, self.n_slow_motor,
            connectivity=0.15, weight_mean=0.4, excitatory=True,
            label="Inter→SlowMotor")

        self.syn_inter_feed = Synapse(n_interneurons, self.n_feeding_motor,
            connectivity=0.15, weight_mean=0.5, excitatory=True,
            label="Inter→FeedMotor")

        # Motorausgabe
        self.mandible_force = 0.0
        self.mandible_speed = 0.0
        self.feeding_active = False

    def step(self, command_input: np.ndarray, taste_input: np.ndarray = None,
             bite_command: float = 0.0, feed_command: float = 0.0,
             dt: float = 0.1) -> dict:
        """
        Ein Zeitschritt.

        Args:
            command_input: Absteigende Kommandos vom Gehirn (n_interneurons,)
            taste_input: Geschmacksinput (n_gustatory,)
            bite_command: Beißbefehl (0-1)
            feed_command: Fütterungsbefehl (0-1)
            dt: Zeitschritt

        Returns:
            Dict mit Motorausgaben
        """
        # Gustatorische Verarbeitung
        if taste_input is not None:
            gust_spikes = self.gustatory.step(taste_input * 3.0, dt)
        else:
            gust_spikes = self.gustatory.step(np.zeros(self.n_gustatory), dt)

        # Interneuronen
        I_inter = command_input[:self.n_interneurons]
        if taste_input is not None:
            I_inter = I_inter + self.syn_gust_inter.transmit(gust_spikes)
        inter_spikes = self.interneurons.step(I_inter, dt)

        # Schnelle Motorneuronen (Beißen)
        I_fast = self.syn_inter_fast.transmit(inter_spikes) + bite_command * 5.0
        fast_spikes = self.fast_motor.step(I_fast, dt)

        # Langsame Motorneuronen (Greifen)
        I_slow = self.syn_inter_slow.transmit(inter_spikes) + (1 - bite_command) * 3.0
        slow_spikes = self.slow_motor.step(I_slow, dt)

        # Fütterungsmotorneuronen
        I_feed = self.syn_inter_feed.transmit(inter_spikes) + feed_command * 4.0
        feed_spikes = self.feeding_motor.step(I_feed, dt)

        # Motorausgabe berechnen
        self.mandible_speed = fast_spikes.sum() / self.n_fast_motor
        self.mandible_force = slow_spikes.sum() / self.n_slow_motor
        self.feeding_active = feed_spikes.sum() > self.n_feeding_motor * 0.3

        return {
            'mandible_speed': self.mandible_speed,
            'mandible_force': self.mandible_force,
            'feeding_active': self.feeding_active,
            'fast_motor_spikes': fast_spikes,
            'slow_motor_spikes': slow_spikes,
            'feeding_motor_spikes': feed_spikes,
        }

    def reset(self):
        self.fast_motor.reset()
        self.slow_motor.reset()
        self.feeding_motor.reset()
        self.gustatory.reset()
        self.interneurons.reset()
        self.mandible_force = 0.0
        self.mandible_speed = 0.0
        self.feeding_active = False
