"""
AntBrain - Zentrales Modul das alle Hirnregionen verbindet.

Signalfluss:
    SENSORIK
    ├── Antennen (Geruch) ──→ Antennallobus ──→ Pilzkörper (Lernen)
    │                                      └──→ Lateralhorn (Instinkt)
    ├── Augen (Visuell)  ──→ Optischer Lobus ──→ Pilzkörper (Collar)
    │                                       └──→ Zentralkomplex (Kompass)
    └── Gustatorisch     ──→ SEG (Geschmack)

    INTEGRATION
    ├── Pilzkörper: Assoziatives Lernen (STDP + Dopamin)
    ├── Lateralhorn: Angeborene Reaktionen
    └── Zentralkomplex: Pfadintegration & Navigation

    MOTORIK
    ├── Zentralkomplex → LAL → Absteigendes NS → Beinsteuerung
    └── SEG → Mandibel-Motorneuronen
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from ant_brain.regions.antennal_lobe import AntennalLobe
from ant_brain.regions.mushroom_body import MushroomBody
from ant_brain.regions.central_complex import CentralComplex
from ant_brain.regions.optic_lobe import OpticLobe
from ant_brain.regions.lateral_horn import LateralHorn
from ant_brain.regions.subesophageal_ganglion import SubesophagealGanglion
from ant_brain.core.synapse import Synapse


@dataclass
class SensoryInput:
    """Sensorische Eingaben für einen Zeitschritt."""
    # Olfaktorisch
    odor_vector: Optional[np.ndarray] = None  # (400,) Geruchsvektor
    odor_type: Optional[str] = None           # Alternativ: Geruchstyp-String
    odor_concentration: float = 0.5

    # Visuell
    visual_field: Optional[np.ndarray] = None  # (600,) Pixelintensitäten
    polarization_angle: float = 0.0            # Polarisiertes Licht (rad)

    # Navigation
    compass_angle: float = 0.0     # Kompassrichtung (rad)
    speed: float = 0.0             # Laufgeschwindigkeit
    angular_velocity: float = 0.0  # Drehgeschwindigkeit (rad/s)

    # Gustatorisch
    taste_input: Optional[np.ndarray] = None  # Geschmack

    # Belohnung
    reward: float = 0.0  # Belohnungssignal (-1 bis +1)

    # Motorkommandos
    bite_command: float = 0.0
    feed_command: float = 0.0


@dataclass
class BrainOutput:
    """Ausgabe des Gehirns für einen Zeitschritt."""
    # Navigation
    steering: np.ndarray = None      # [links, rechts] Steuerung
    heading: float = 0.0
    home_vector: np.ndarray = None
    home_distance: float = 0.0

    # Verhalten
    behavioral_state: str = "idle"   # idle, foraging, alarm, homing, feeding
    alarm_level: float = 0.0
    attraction_level: float = 0.0

    # Motorik
    mandible_speed: float = 0.0
    mandible_force: float = 0.0
    feeding_active: bool = False

    # Lernen
    kc_sparseness: float = 0.0
    learned_value: np.ndarray = None

    def __post_init__(self):
        if self.steering is None:
            self.steering = np.zeros(2)
        if self.home_vector is None:
            self.home_vector = np.zeros(2)
        if self.learned_value is None:
            self.learned_value = np.zeros(21)


class Neuromodulator:
    """
    Neuromodulationssystem.

    Kontrolliert globale Gehirnzustände über biogene Amine:
    - Dopamin: Belohnungslernen, Gedächtniskonsolidierung
    - Octopamin: Appetitives Lernen, Nestkameraden-Erkennung
    - Serotonin: Aggressionsmodulation, sozialer Status
    - Tyramin: Lokomotion, sensorisches Gating
    """

    def __init__(self):
        self.levels = {
            'dopamine': 1.0,
            'octopamine': 1.0,
            'serotonin': 1.0,
            'tyramine': 1.0,
        }
        # Zeitkonstanten für Auf-/Abbau
        self.tau_rise = 50.0    # ms
        self.tau_decay = 500.0  # ms

    def update(self, reward: float, social_context: float = 0.0,
               threat: float = 0.0, dt: float = 0.1):
        """Neuromodulatorlevel basierend auf Kontext updaten."""
        # Dopamin: steigt bei Belohnung
        da_target = 1.0 + reward * 0.5
        tau = self.tau_rise if da_target > self.levels['dopamine'] else self.tau_decay
        self.levels['dopamine'] += (da_target - self.levels['dopamine']) * dt / tau

        # Octopamin: steigt beim Futtersuchen
        oa_target = 1.0 + max(0, reward) * 0.3
        tau = self.tau_rise if oa_target > self.levels['octopamine'] else self.tau_decay
        self.levels['octopamine'] += (oa_target - self.levels['octopamine']) * dt / tau

        # Serotonin: steigt bei Bedrohung (Aggression)
        ht_target = 1.0 + threat * 0.5
        tau = self.tau_rise if ht_target > self.levels['serotonin'] else self.tau_decay
        self.levels['serotonin'] += (ht_target - self.levels['serotonin']) * dt / tau

        # Tyramin: Lokomotionsmodulation
        self.levels['tyramine'] = np.clip(self.levels['tyramine'], 0.5, 1.5)

    def apply_to_region(self, neurons):
        """Neuromodulation auf eine Neuronengruppe anwenden."""
        for nt, level in self.levels.items():
            if hasattr(neurons, 'set_modulation'):
                neurons.set_modulation(nt, level)


class AntBrain:
    """
    Vollständiges Ameisengehirn (~250.000 Neuronen).

    Verbindet alle Hirnregionen und koordiniert den Informationsfluss.
    """

    def __init__(self, config: dict = None):
        config = config or {}

        # --- Hirnregionen instanziieren ---
        self.antennal_lobe = AntennalLobe(
            n_glomeruli=config.get('n_glomeruli', 400),
        )

        # Tatsächliche Anzahl m-ALT und l-ALT PNs aus AL-Masken
        n_malt = int(self.antennal_lobe.malt_mask.sum())
        n_lalt = int(self.antennal_lobe.lalt_mask.sum())

        self.mushroom_body = MushroomBody(
            n_kenyon_cells=config.get('n_kenyon_cells', 80000),
            n_pn_input=n_malt,
        )

        self.optic_lobe = OpticLobe(
            n_ommatidia=config.get('n_ommatidia', 600),
        )

        self.central_complex = CentralComplex(
            n_columns=config.get('cx_columns', 16),
        )

        n_glom = config.get('n_glomeruli', 400)
        pns_per_glom = self.antennal_lobe.pns_per_glom
        self.lateral_horn = LateralHorn(
            n_neurons=config.get('lh_neurons', 3000),
            n_pn_input=n_lalt,
            pns_per_glom=pns_per_glom,
            n_glomeruli=n_glom,
        )

        self.seg = SubesophagealGanglion()

        # --- Neuromodulation ---
        self.neuromodulator = Neuromodulator()

        # --- Inter-regionale Verbindungen ---
        # OL → CX (Kompass via anteriorer optischer Tuberkel)
        self.syn_ol_cx = Synapse(
            self.optic_lobe.n_dra,
            self.central_complex.ring_attractor.n_total,
            connectivity=0.2,
            weight_mean=0.7,
            excitatory=True,
            label="OL→CX_compass"
        )

        # MB → Verhaltensentscheidung (prämotorisch)
        # LH → Verhaltensentscheidung

        # --- Zustand ---
        self.time_ms = 0.0
        self.dt = 0.5  # Zeitschritt in ms
        self.behavioral_state = "idle"
        self.step_count = 0

        # Neuronenzählung
        self._count_neurons()

    def _count_neurons(self):
        """Zähle die Gesamtanzahl der Neuronen im Modell."""
        counts = {
            'Antennallobus': sum(
                g.n_lns + g.n_pns for g in self.antennal_lobe.glomeruli
            ),
            'Pilzkörper': (
                self.mushroom_body.n_kc +
                self.mushroom_body.n_mbon +
                self.mushroom_body.n_dan
            ),
            'Optischer Lobus': (
                self.optic_lobe.n_lamina +
                self.optic_lobe.n_medulla +
                self.optic_lobe.n_lobula +
                self.optic_lobe.n_dra
            ),
            'Zentralkomplex': (
                self.central_complex.n_pb +
                self.central_complex.ring_attractor.n_total +
                self.central_complex.n_fb +
                self.central_complex.n_noduli
            ),
            'Lateralhorn': (
                self.lateral_horn.n_neurons +
                self.lateral_horn.N_CHANNELS
            ),
            'SEG': (
                self.seg.n_fast_motor +
                self.seg.n_slow_motor +
                self.seg.n_feeding_motor +
                self.seg.n_gustatory +
                self.seg.n_interneurons
            ),
        }
        self.neuron_counts = counts
        self.total_neurons = sum(counts.values())

    def step(self, sensory: SensoryInput) -> BrainOutput:
        """
        Ein Simulationsschritt des gesamten Gehirns.

        Args:
            sensory: Sensorische Eingaben

        Returns:
            BrainOutput mit allen Ausgaben
        """
        dt = self.dt
        output = BrainOutput()

        # --- 1. Neuromodulation updaten ---
        threat = 0.0
        self.neuromodulator.update(sensory.reward, threat=threat, dt=dt)

        # --- 2. Olfaktorische Verarbeitung (Antennallobus) ---
        if sensory.odor_vector is not None:
            odor_vec = sensory.odor_vector
        elif sensory.odor_type is not None:
            odor_vec = self.antennal_lobe.create_odor_vector(
                sensory.odor_type, sensory.odor_concentration)
        else:
            odor_vec = np.zeros(self.antennal_lobe.n_glomeruli)

        al_result = self.antennal_lobe.step(odor_vec, dt)

        # --- 3. Visuelle Verarbeitung (Optischer Lobus) ---
        if sensory.visual_field is not None:
            vis_input = sensory.visual_field
        else:
            vis_input = np.zeros(self.optic_lobe.n_ommatidia)

        ol_result = self.optic_lobe.step(
            vis_input, sensory.polarization_angle, dt)

        # --- 4. Lateralhorn (angeborene Reaktionen) ---
        lh_result = self.lateral_horn.step(al_result['lalt_spikes'], dt)

        # --- 5. Pilzkörper (Lernen und Integration) ---
        # Visueller Input: OL lobula spikes direkt als visuelle Eingabe
        # (MB integriert visuellen Input intern über syn_vis_kc)
        vis_spikes_for_mb = ol_result['lobula_spikes'][:self.mushroom_body.n_visual]

        mb_result = self.mushroom_body.step(
            al_result['malt_spikes'],
            visual_spikes=vis_spikes_for_mb,
            reward_signal=sensory.reward,
            dt=dt
        )

        # --- 6. Zentralkomplex (Navigation) ---
        cx_result = self.central_complex.step(
            sensory.compass_angle,
            sensory.speed,
            sensory.angular_velocity,
            dt
        )

        # --- 7. Subesophagealganglion (Motorsteuerung) ---
        seg_command = np.zeros(self.seg.n_interneurons)
        # LH-Alarm → SEG (Beißreflex)
        if lh_result['dominant_response'] == LateralHorn.CHANNEL_ALARM:
            seg_command[:50] = 3.0  # Alarm-Kommando

        seg_result = self.seg.step(
            seg_command,
            taste_input=sensory.taste_input,
            bite_command=sensory.bite_command,
            feed_command=sensory.feed_command,
            dt=dt
        )

        # --- 8. Verhaltenszustand bestimmen ---
        self._update_behavioral_state(lh_result, mb_result, cx_result)

        # --- 9. Ausgabe zusammenstellen ---
        output.steering = cx_result['steering']
        output.heading = cx_result['heading']
        output.home_vector = cx_result['home_vector']
        output.home_distance = cx_result['home_distance']

        output.behavioral_state = self.behavioral_state
        output.alarm_level = lh_result['channel_activity'][LateralHorn.CHANNEL_ALARM]
        output.attraction_level = lh_result['channel_activity'][LateralHorn.CHANNEL_ATTRACTION]

        output.mandible_speed = seg_result['mandible_speed']
        output.mandible_force = seg_result['mandible_force']
        output.feeding_active = seg_result['feeding_active']

        output.kc_sparseness = mb_result['kc_sparseness']
        output.learned_value = mb_result['mbon_values']

        # Zeit aktualisieren
        self.time_ms += dt
        self.step_count += 1

        return output

    def _update_behavioral_state(self, lh_result, mb_result, cx_result):
        """Verhaltenszustand basierend auf neuronaler Aktivität bestimmen."""
        alarm = lh_result['channel_activity'][LateralHorn.CHANNEL_ALARM]
        attraction = lh_result['channel_activity'][LateralHorn.CHANNEL_ATTRACTION]

        if alarm > 0.01 and alarm > attraction * 1.5:
            self.behavioral_state = "alarm"
        elif attraction > 0.005:
            self.behavioral_state = "foraging"
        elif cx_result['home_distance'] > 5.0:
            self.behavioral_state = "homing"
        else:
            self.behavioral_state = "idle"

    def run(self, sensory_sequence: list[SensoryInput],
            callback=None) -> list[BrainOutput]:
        """
        Simulation über mehrere Zeitschritte.

        Args:
            sensory_sequence: Liste von SensoryInput pro Zeitschritt
            callback: Optionale Funktion(step, output) für jeden Schritt

        Returns:
            Liste von BrainOutput
        """
        outputs = []
        for i, sensory in enumerate(sensory_sequence):
            output = self.step(sensory)
            outputs.append(output)
            if callback:
                callback(i, output)
        return outputs

    def get_statistics(self) -> dict:
        """Aktuelle Statistiken des Gehirns."""
        return {
            'total_neurons': self.total_neurons,
            'neuron_counts': self.neuron_counts,
            'time_ms': self.time_ms,
            'steps': self.step_count,
            'behavioral_state': self.behavioral_state,
            'neuromodulation': dict(self.neuromodulator.levels),
        }

    def reset(self):
        """Gesamtes Gehirn zurücksetzen."""
        self.antennal_lobe.reset()
        self.mushroom_body.reset()
        self.optic_lobe.reset()
        self.central_complex.reset()
        self.lateral_horn.reset()
        self.seg.reset()
        self.time_ms = 0.0
        self.step_count = 0
        self.behavioral_state = "idle"

    def __repr__(self):
        lines = [f"AntBrain (Σ {self.total_neurons:,} Neuronen)"]
        for region, count in self.neuron_counts.items():
            pct = count / self.total_neurons * 100
            lines.append(f"  {region}: {count:,} ({pct:.1f}%)")
        return "\n".join(lines)
