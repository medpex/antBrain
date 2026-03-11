"""
Antennensensor - Olfaktorischer Input.

Simuliert die ~50.000-60.000 olfaktorischen Rezeptorneuronen (ORNs)
auf jeder Antenne der Ameise.

Funktionen:
- Pheromon-Detektion (Alarm, Trail, Nestmate)
- Nahrungserkennung
- Bilateral-Vergleich für Osmotropotaxis
"""

import numpy as np


class Antenna:
    """
    Simulierte Ameisenantenne mit olfaktorischen Rezeptoren.

    Wandelt Umgebungsgerüche in Aktivierungsmuster um,
    die an den Antennallobus weitergeleitet werden.
    """

    def __init__(self, n_receptor_types: int = 400, side: str = "left"):
        """
        Args:
            n_receptor_types: Anzahl Rezeptortypen (≈ Glomeruli-Anzahl)
            side: 'left' oder 'right' für bilaterale Verarbeitung
        """
        self.n_types = n_receptor_types
        self.side = side

        # Rezeptorsensitivitäten (zufällig initialisiert, aber konsistent)
        rng = np.random.RandomState(hash(side) % 2**31)
        self.sensitivity = rng.uniform(0.5, 1.5, n_receptor_types)

        # Adaptationsrate (Gewöhnung bei langanhaltenden Gerüchen)
        self.adaptation = np.ones(n_receptor_types)
        self.tau_adaptation = 5000.0  # ms

        # Rauschparameter
        self.noise_level = 0.02

    def sense(self, odor_sources: list[dict], position: np.ndarray,
              wind_direction: float = 0.0, dt: float = 0.1) -> np.ndarray:
        """
        Geruchswahrnehmung.

        Args:
            odor_sources: Liste von Geruchsquellen
                [{'position': [x,y], 'type': 'trail', 'concentration': 0.5,
                  'vector': np.ndarray(400)}]
            position: Position der Ameise [x, y]
            wind_direction: Windrichtung in rad
            dt: Zeitschritt

        Returns:
            Aktivierungsvektor (n_types,) für den Antennallobus
        """
        activation = np.zeros(self.n_types)

        for source in odor_sources:
            src_pos = np.array(source.get('position', [0, 0]))
            distance = np.linalg.norm(position - src_pos) + 0.01

            # Konzentration nimmt mit Distanz ab (exponential decay)
            base_conc = source.get('concentration', 0.5)
            conc = base_conc * np.exp(-distance * 0.05)

            # Wind beeinflusst Ausbreitung
            if 'position' in source:
                angle_to_source = np.arctan2(
                    src_pos[1] - position[1],
                    src_pos[0] - position[0])
                wind_factor = 1 + 0.3 * np.cos(wind_direction - angle_to_source)
                conc *= wind_factor

            # Bilateral-Offset (linke vs rechte Antenne)
            if self.side == 'left':
                conc *= 1.0 + 0.05 * np.cos(wind_direction)
            else:
                conc *= 1.0 - 0.05 * np.cos(wind_direction)

            # Geruchsvektor anwenden
            if 'vector' in source:
                activation += source['vector'] * conc
            else:
                activation += conc * 0.1  # Unspezifischer Hintergrund

        # Rezeptorsensitivität
        activation *= self.sensitivity

        # Adaptation (zeitnormalisiert)
        # Bei Stimulation: exponentielle Abschwächung
        # Ohne Stimulation: Erholung Richtung 1.0
        decay = np.exp(-dt / self.tau_adaptation)
        self.adaptation = np.where(
            activation > 0.1,
            self.adaptation * decay,               # Adaptation bei Stimulation
            self.adaptation + (1.0 - self.adaptation) * dt / self.tau_adaptation  # Erholung
        )
        self.adaptation = np.clip(self.adaptation, 0.1, 1.0)
        activation *= self.adaptation

        # Rauschen
        activation += np.abs(np.random.normal(0, self.noise_level, self.n_types))

        return np.clip(activation, 0, 1)

    def reset(self):
        self.adaptation[:] = 1.0
