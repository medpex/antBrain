#!/usr/bin/env python3
"""
Demo: Futtersuch-Simulation einer Ameise.

Zeigt das Zusammenspiel aller Hirnregionen:
1. Ameise startet am Nest
2. Sucht nach Nahrung (Geruchsgradienten folgen)
3. Lernt Nahrungsort über Pilzkörper (STDP)
4. Navigiert zurück zum Nest (Pfadintegration im CX)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ant_brain.core.brain import AntBrain, SensoryInput
from ant_brain.sensors.antenna import Antenna
from ant_brain.sensors.eye import CompoundEye
from ant_brain.actuators.locomotion import LocomotionController


def run_foraging_simulation(duration_ms: float = 5000, dt: float = 0.5):
    """
    Futtersuch-Simulation.

    Eine Ameise mit vollständigem Gehirn sucht nach Nahrung
    und navigiert zurück zum Nest.
    """
    print("=" * 60)
    print("AMEISENGEHIRN-SIMULATION: Futtersuche")
    print("=" * 60)

    # --- Gehirn initialisieren ---
    # Reduzierte Größe für schnellere Demo
    config = {
        'n_glomeruli': 50,       # Reduziert von 400
        'n_kenyon_cells': 5000,  # Reduziert von 80000
        'n_ommatidia': 100,      # Reduziert von 600
        'cx_columns': 8,
        'lh_neurons': 500,
    }

    print("\nGehirn wird initialisiert...")
    brain = AntBrain(config)
    print(brain)
    print()

    # --- Sensoren und Aktuatoren ---
    left_antenna = Antenna(n_receptor_types=50, side="left")
    right_antenna = Antenna(n_receptor_types=50, side="right")
    left_eye = CompoundEye(n_ommatidia=100, side="left")
    right_eye = CompoundEye(n_ommatidia=100, side="right")
    locomotion = LocomotionController()

    # --- Umgebung ---
    nest_pos = np.array([50.0, 50.0])
    food_pos = np.array([30.0, 35.0])

    # Nahrungsgeruchsvektor
    food_odor = np.zeros(50)
    food_odor[5:10] = 1.0  # Nahrungsgeruch aktiviert bestimmte Glomeruli

    # Trail-Pheromonpositionen
    trail_points = [
        np.array([45.0, 48.0]),
        np.array([40.0, 45.0]),
        np.array([35.0, 40.0]),
        np.array([30.0, 35.0]),
    ]

    # --- Simulation ---
    n_steps = int(duration_ms / dt)
    print(f"Simuliere {n_steps} Schritte ({duration_ms}ms bei dt={dt}ms)...")
    print()

    # Aufzeichnung
    positions = []
    headings = []
    behavioral_states = []
    alarm_levels = []
    attraction_levels = []
    sparseness_history = []
    home_distances = []

    # Startzustand
    locomotion.position = nest_pos.copy()
    locomotion.heading = np.random.uniform(0, 2 * np.pi)

    phase = "outbound"  # outbound → foraging → homing

    for step in range(n_steps):
        pos = locomotion.position
        heading = locomotion.heading

        # --- Sensorische Eingaben erzeugen ---
        # Geruchsquellen in der Umgebung
        odor_sources = [
            {
                'position': food_pos,
                'type': 'food',
                'concentration': 0.8,
                'vector': food_odor,
            }
        ]

        # Trail-Pheromon
        for tp in trail_points:
            trail_vec = np.zeros(50)
            trail_vec[1:3] = 1.0
            odor_sources.append({
                'position': tp,
                'type': 'trail',
                'concentration': 0.3,
                'vector': trail_vec,
            })

        # Antennen-Wahrnehmung
        left_smell = left_antenna.sense(odor_sources, pos, heading, dt)
        right_smell = right_antenna.sense(odor_sources, pos, heading, dt)

        # Bilateraler Mittelwert
        odor_input = (left_smell + right_smell) / 2.0

        # Visuelle Eingabe
        landmarks = [
            {'angle': 0.0, 'distance': np.linalg.norm(pos - nest_pos),
             'size': 5.0, 'brightness': 0.8},
        ]
        visual_input = left_eye.process_scene(landmarks=landmarks, heading=heading)

        # Belohnungssignal
        dist_to_food = np.linalg.norm(pos - food_pos)
        reward = 0.0
        if dist_to_food < 3.0 and phase == "outbound":
            reward = 1.0  # Nahrung gefunden!
            phase = "homing"
            print(f"  Schritt {step}: NAHRUNG GEFUNDEN! Beginne Heimkehr.")

        # Kompass (Sonne)
        compass = heading + np.random.normal(0, 0.05)

        # Geschwindigkeit
        speed = 0.5

        # --- Gehirn-Schritt ---
        sensory = SensoryInput(
            odor_vector=odor_input,
            visual_field=visual_input,
            polarization_angle=compass * 0.5,
            compass_angle=compass,
            speed=speed,
            angular_velocity=locomotion.angular_velocity,
            reward=reward,
        )

        output = brain.step(sensory)

        # --- Lokomotion ---
        # Steering: Mischung aus CX-Steering und Geruchsgradienten
        bilateral_diff = right_smell.sum() - left_smell.sum()
        chemotaxis_steering = np.array([
            max(0, -bilateral_diff * 0.5),
            max(0, bilateral_diff * 0.5)
        ])

        combined_steering = output.steering * 0.6 + chemotaxis_steering * 0.4

        if phase == "homing":
            # Im Homing-Modus: CX-Pfadintegration dominiert
            combined_steering = output.steering * 0.9 + chemotaxis_steering * 0.1

        loco_result = locomotion.step(combined_steering, speed, dt)

        # --- Aufzeichnung ---
        positions.append(pos.copy())
        headings.append(heading)
        behavioral_states.append(output.behavioral_state)
        alarm_levels.append(output.alarm_level)
        attraction_levels.append(output.attraction_level)
        sparseness_history.append(output.kc_sparseness)
        home_distances.append(output.home_distance)

        # Fortschritt
        if step % (n_steps // 10) == 0:
            print(f"  Schritt {step}/{n_steps} | "
                  f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}) | "
                  f"Zustand: {output.behavioral_state} | "
                  f"KC-Sparseness: {output.kc_sparseness:.4f} | "
                  f"Heimdistanz: {output.home_distance:.2f}")

    # --- Ergebnisse ---
    print("\n" + "=" * 60)
    print("ERGEBNIS")
    print("=" * 60)
    stats = brain.get_statistics()
    print(f"Gesamtneuronen: {stats['total_neurons']:,}")
    for region, count in stats['neuron_counts'].items():
        print(f"  {region}: {count:,}")
    print(f"\nNeuromodulation:")
    for nm, level in stats['neuromodulation'].items():
        print(f"  {nm}: {level:.3f}")
    print(f"\nEndposition: ({locomotion.position[0]:.1f}, {locomotion.position[1]:.1f})")
    print(f"Endheading: {np.degrees(locomotion.heading):.1f}°")

    # --- Visualisierung ---
    plot_results(positions, headings, behavioral_states, alarm_levels,
                attraction_levels, sparseness_history, home_distances,
                nest_pos, food_pos, trail_points, dt)

    return brain, positions


def plot_results(positions, headings, states, alarm, attraction,
                sparseness, home_dist, nest_pos, food_pos, trail_points, dt):
    """Erstelle Visualisierung der Simulation."""
    positions = np.array(positions)
    time_ms = np.arange(len(positions)) * dt

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Pfad der Ameise
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    colors = np.linspace(0, 1, len(positions))
    ax1.scatter(positions[:, 0], positions[:, 1], c=colors, cmap='viridis',
               s=1, alpha=0.5)
    ax1.plot(positions[0, 0], positions[0, 1], 'g^', markersize=12,
            label='Start')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=12,
            label='Ende')
    ax1.plot(nest_pos[0], nest_pos[1], 'bo', markersize=15, label='Nest')
    ax1.plot(food_pos[0], food_pos[1], 'r*', markersize=15, label='Nahrung')
    for tp in trail_points:
        ax1.plot(tp[0], tp[1], 'c.', markersize=8)
    ax1.set_xlabel('X (cm)')
    ax1.set_ylabel('Y (cm)')
    ax1.set_title('Pfad der Ameise')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 2. Verhaltenszustand
    ax2 = fig.add_subplot(gs[0, 2])
    state_map = {'idle': 0, 'foraging': 1, 'alarm': 2, 'homing': 3, 'feeding': 4}
    state_nums = [state_map.get(s, 0) for s in states]
    ax2.plot(time_ms, state_nums, 'b-', linewidth=0.5)
    ax2.set_yticks(list(state_map.values()))
    ax2.set_yticklabels(list(state_map.keys()), fontsize=8)
    ax2.set_xlabel('Zeit (ms)')
    ax2.set_title('Verhaltenszustand')
    ax2.grid(True, alpha=0.3)

    # 3. Alarm & Attraktion
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(time_ms, alarm, 'r-', alpha=0.7, label='Alarm', linewidth=0.5)
    ax3.plot(time_ms, attraction, 'g-', alpha=0.7, label='Attraktion', linewidth=0.5)
    ax3.set_xlabel('Zeit (ms)')
    ax3.set_ylabel('Aktivität')
    ax3.set_title('Lateralhorn-Kanäle')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. KC-Sparseness
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time_ms, sparseness, 'purple', linewidth=0.5)
    ax4.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Ziel (5%)')
    ax4.set_xlabel('Zeit (ms)')
    ax4.set_ylabel('Aktivierungsdichte')
    ax4.set_title('KC-Sparseness (Pilzkörper)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Heimdistanz
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time_ms, home_dist, 'orange', linewidth=0.5)
    ax5.set_xlabel('Zeit (ms)')
    ax5.set_ylabel('Distanz')
    ax5.set_title('Heimvektor-Distanz (CX)')
    ax5.grid(True, alpha=0.3)

    # 6. Heading
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(time_ms, np.degrees(headings), 'b-', linewidth=0.5)
    ax6.set_xlabel('Zeit (ms)')
    ax6.set_ylabel('Richtung (°)')
    ax6.set_title('Kopfrichtung')
    ax6.grid(True, alpha=0.3)

    plt.savefig('/root/neuronal/simulations/foraging_result.png', dpi=150,
               bbox_inches='tight')
    print("\nVisualisierung gespeichert: simulations/foraging_result.png")


if __name__ == "__main__":
    brain, positions = run_foraging_simulation(duration_ms=3000, dt=0.5)
