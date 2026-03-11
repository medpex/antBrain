"""
Lokomotionssystem - Beinsteuerung.

Die eigentliche Lokomotion wird von Central Pattern Generators (CPGs)
in den Thorakalganglien erzeugt. Das Gehirn gibt nur Richtung
und Geschwindigkeit vor.

Implementiert:
- Tripod-Gang (3 Beine gleichzeitig)
- Geschwindigkeitssteuerung
- Drehsteuerung basierend auf CX-Steering
"""

import numpy as np


class LocomotionController:
    """
    Lokomotionssteuerung basierend auf CPGs.

    Wandelt Steering-Signale vom Zentralkomplex in
    Beinbewegungen um.
    """

    def __init__(self):
        # CPG-Phase (Tripod-Gang)
        self.phase = 0.0
        self.cpg_frequency = 10.0  # Hz (Grundfrequenz)

        # Aktuelle Geschwindigkeit und Richtung
        self.velocity = np.zeros(2)  # [vx, vy]
        self.angular_velocity = 0.0
        self.speed = 0.0

        # Position und Orientierung
        self.position = np.zeros(2)
        self.heading = 0.0

        # Beinzustände (6 Beine: L1,L2,L3,R1,R2,R3)
        self.leg_phases = np.array([0, np.pi, 0, np.pi, 0, np.pi])
        self.leg_names = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']

        # Maximale Geschwindigkeit (cm/s, typisch für Ameise)
        self.max_speed = 3.0  # ~3 cm/s
        self.max_turn_rate = 2.0  # rad/s

    def step(self, steering: np.ndarray, speed_command: float = 0.5,
             dt: float = 0.1) -> dict:
        """
        Ein Zeitschritt der Lokomotion.

        Args:
            steering: [links, rechts] Steuersignal vom CX
            speed_command: Geschwindigkeitsbefehl (0-1)
            dt: Zeitschritt in ms

        Returns:
            Dict mit Position, Geschwindigkeit, Beinzuständen
        """
        dt_s = dt / 1000.0  # ms → s

        # Drehung aus Steering ableiten
        turn = (steering[1] - steering[0]) * self.max_turn_rate
        self.angular_velocity = turn

        # Heading updaten
        self.heading += turn * dt_s
        self.heading %= (2 * np.pi)

        # Geschwindigkeit
        self.speed = speed_command * self.max_speed

        # Position updaten
        self.position[0] += self.speed * np.cos(self.heading) * dt_s
        self.position[1] += self.speed * np.sin(self.heading) * dt_s

        # CPG-Phase updaten
        freq = self.cpg_frequency * speed_command
        self.phase += 2 * np.pi * freq * dt_s
        self.phase %= (2 * np.pi)

        # Beinphasen updaten (Tripod-Gang)
        self.leg_phases = (self.leg_phases + 2 * np.pi * freq * dt_s) % (2 * np.pi)

        # Beinzustände (0 = Stance, 1 = Swing)
        leg_states = (np.sin(self.leg_phases) > 0).astype(float)

        return {
            'position': self.position.copy(),
            'heading': self.heading,
            'speed': self.speed,
            'angular_velocity': self.angular_velocity,
            'leg_states': leg_states,
            'leg_names': self.leg_names,
        }

    def reset(self):
        self.position[:] = 0
        self.heading = 0
        self.speed = 0
        self.angular_velocity = 0
        self.phase = 0
        self.leg_phases = np.array([0, np.pi, 0, np.pi, 0, np.pi])
