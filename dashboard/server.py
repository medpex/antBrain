#!/usr/bin/env python3
"""
Live-Dashboard-Server für die Ameisengehirn-Simulation.

Verwendet Python's eingebauten HTTP-Server mit SSE für Live-Updates.
Startet eine vollständige Gehirn-Simulation mit biologisch realistischen Parametern.
"""

import sys
import os
import json
import time
import threading
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ant_brain.core.brain import AntBrain, SensoryInput
from ant_brain.sensors.antenna import Antenna
from ant_brain.sensors.eye import CompoundEye
from ant_brain.actuators.locomotion import LocomotionController

PORT = 8050


def _json_default(obj):
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class Simulation:
    """Simulation mit vollem Ameisengehirn."""

    def __init__(self):
        self.running = False
        self.paused = False
        self.speed = 1.0         # Simulationsgeschwindigkeit
        self.dt = 0.5            # ms pro Schritt
        self.step_count = 0
        self.sim_time_ms = 0.0

        # Injektionen vom Dashboard
        self.inject_odor = None       # z.B. 'alarm', 'food', 'trail'
        self.inject_concentration = 0.0
        self.inject_reward = 0.0

        # Aktueller Zustand (thread-safe gelesen)
        self.state = {}
        self.lock = threading.Lock()

        # --- Biologisches Verhalten ---
        self.carrying_food = False        # Ameise trägt Futter
        self.carried_food_type = None     # Typ des getragenen Futters
        self.food_collected = 0           # Gesamtzahl gesammelter Nahrung
        self.food_at_nest = 0             # Nahrung im Nest abgelegt
        self.satiation = 0.5             # Sättigungsgrad (0=hungrig, 1=satt)
        self.energy = 1.0                # Energielevel (sinkt beim Laufen)

        # Pheromonsystem (Grid-basiert, 100x100 Arena)
        self.pheromone_trail = np.zeros((100, 100))    # Spur-Pheromon
        self.pheromone_alarm = np.zeros((100, 100))    # Alarm-Pheromon
        self.pheromone_food = np.zeros((100, 100))     # Futter-Pheromon
        self.pheromone_decay = 0.9995    # Langsamer Zerfall pro Schritt

        # Nahrungsquellen mit Menge
        self.food_amounts = {}  # Index → verbleibende Menge

        # Beinzustände
        self.leg_states = [0, 1, 0, 1, 0, 1]

        # Vorberechnete Geruchsvektoren (Performance)
        self._food_vec = np.zeros(400)
        self._food_vec[35:55] = 1.0
        self._trail_vec = np.zeros(400)
        self._trail_vec[5:15] = 1.0
        self._nest_vec = np.zeros(400)
        self._nest_vec[15:35] = 1.0
        self._alarm_vec = np.zeros(400)
        self._alarm_vec[0:5] = 1.0

        self._init_brain()

    def _init_brain(self):
        print("Initialisiere reales Ameisengehirn...")
        t0 = time.time()

        self.brain = AntBrain({
            'n_glomeruli': 400,
            'n_kenyon_cells': 50000,
            'n_ommatidia': 600,
            'cx_columns': 16,
            'lh_neurons': 3000,
        })

        self.left_antenna = Antenna(n_receptor_types=400, side="left")
        self.right_antenna = Antenna(n_receptor_types=400, side="right")
        self.left_eye = CompoundEye(n_ommatidia=600, side="left")
        self.right_eye = CompoundEye(n_ommatidia=600, side="right")
        self.locomotion = LocomotionController()

        # Arena
        self.nest_pos = np.array([50.0, 50.0])
        self.food_sources = [
            {'pos': np.array([25.0, 30.0]), 'type': 'sugar', 'conc': 0.8, 'amount': 20.0},
            {'pos': np.array([75.0, 70.0]), 'type': 'protein', 'conc': 0.6, 'amount': 15.0},
            {'pos': np.array([20.0, 75.0]), 'type': 'sugar', 'conc': 0.4, 'amount': 10.0},
        ]
        # Initialisiere Nahrungsmengen
        for i, fs in enumerate(self.food_sources):
            self.food_amounts[i] = fs['amount']

        # Position
        self.locomotion.position = self.nest_pos.copy()
        self.locomotion.heading = np.random.uniform(0, 2 * np.pi)

        # Verhaltens-Timer
        self.food_pickup_timer = 0      # Zeitschritte zum Aufheben
        self.food_deposit_timer = 0     # Zeitschritte zum Ablegen
        self.exploration_timer = 0      # Exploration-Intervall

        # Historie
        self.position_history = [self.nest_pos.copy().tolist()]
        self.max_history = 2000

        # Spike-Raster (rolling window)
        self.spike_window = 200   # Letzte N Schritte
        self.region_spike_rates = {
            'AL': [], 'MB_KC': [], 'MB_MBON': [],
            'OL': [], 'CX': [], 'LH': [], 'SEG': []
        }

        t1 = time.time()
        print(f"Gehirn initialisiert in {t1-t0:.1f}s")
        print(self.brain)

    def start(self):
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        error_count = 0
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            t0 = time.time()
            try:
                self._step()
                error_count = 0  # Reset bei Erfolg
            except Exception as e:
                error_count += 1
                print(f"[FEHLER] Simulationsschritt {self.step_count}: {e}", flush=True)
                if error_count > 10:
                    print("[FEHLER] Zu viele Fehler, pausiere 1s...", flush=True)
                    time.sleep(1.0)
                    error_count = 0
                continue
            elapsed = time.time() - t0

            # Geschwindigkeitsregelung
            target_dt = 0.02 / max(self.speed, 0.1)
            sleep_time = max(0, target_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _step(self):
        pos = self.locomotion.position
        heading = self.locomotion.heading

        # === PHEROMON-ZERFALL ===
        self.pheromone_trail *= self.pheromone_decay
        self.pheromone_alarm *= self.pheromone_decay
        self.pheromone_food *= 0.9998  # Futterpheromon zerfällt langsamer

        # === ENERGIE-SYSTEM ===
        # Energie sinkt beim Laufen, steigt beim Fressen
        self.energy = max(0.0, self.energy - 0.00002)
        if self.carrying_food:
            self.energy = max(0.0, self.energy - 0.00001)  # Extra-Kosten beim Tragen

        # === GERUCHSQUELLEN AUFBAUEN ===
        odor_sources = []

        # Nahrungsquellen (nur wenn noch Nahrung vorhanden)
        for i, fs in enumerate(self.food_sources):
            amount = self.food_amounts.get(i, 0)
            if amount > 0:
                conc = fs['conc'] * min(1.0, amount / 5.0)
                odor_sources.append({
                    'position': fs['pos'], 'concentration': conc,
                    'vector': self._food_vec,
                })

        # Pheromon-Spur als Geruchsquelle (aggregiert statt einzelne Grid-Zellen)
        gx, gy = int(np.clip(pos[0], 0, 99)), int(np.clip(pos[1], 0, 99))
        x_lo, x_hi = max(0, gx - 3), min(100, gx + 4)
        y_lo, y_hi = max(0, gy - 3), min(100, gy + 4)
        trail_patch = self.pheromone_trail[x_lo:x_hi, y_lo:y_hi]
        if trail_patch.max() > 0.01:
            # Finde stärksten Punkt und aggregiere als eine Quelle
            local_max = trail_patch.max()
            max_idx = np.unravel_index(trail_patch.argmax(), trail_patch.shape)
            odor_sources.append({
                'position': np.array([x_lo + max_idx[0] + 0.5, y_lo + max_idx[1] + 0.5]),
                'concentration': min(float(local_max), 0.8),
                'vector': self._trail_vec,
            })

        # Nest-Pheromon (immer vorhanden, stärker in der Nähe)
        nest_dist = np.linalg.norm(pos - self.nest_pos)
        if nest_dist < 20.0:
            odor_sources.append({
                'position': self.nest_pos,
                'concentration': max(0.1, 1.0 - nest_dist / 20.0),
                'vector': self._nest_vec,
            })

        # Injektionen vom Dashboard
        inject_type = self.inject_odor
        inject_conc = self.inject_concentration
        if inject_type:
            vecs = {'alarm': self._alarm_vec, 'food': self._food_vec,
                    'trail': self._trail_vec, 'nestmate': self._nest_vec}
            inj_vec = vecs.get(inject_type, self._food_vec)
            odor_sources.append({
                'position': pos.copy(), 'concentration': inject_conc,
                'vector': inj_vec,
            })

        # === ANTENNEN ===
        left_smell = self.left_antenna.sense(odor_sources, pos, heading, self.dt)
        right_smell = self.right_antenna.sense(odor_sources, pos, heading, self.dt)
        odor_input = (left_smell + right_smell) / 2.0

        # === AUGEN ===
        landmarks = [
            {'angle': np.arctan2(self.nest_pos[1] - pos[1], self.nest_pos[0] - pos[0]),
             'distance': nest_dist, 'size': 5.0, 'brightness': 0.8},
        ]
        for i, fs in enumerate(self.food_sources):
            if self.food_amounts.get(i, 0) > 0:
                landmarks.append({
                    'angle': np.arctan2(fs['pos'][1] - pos[1], fs['pos'][0] - pos[0]),
                    'distance': np.linalg.norm(pos - fs['pos']),
                    'size': 2.0, 'brightness': 0.5,
                })
        visual_input = self.left_eye.process_scene(landmarks=landmarks, heading=heading)

        # === BIOLOGISCHES VERHALTEN: FUTTER-INTERAKTION ===
        reward = self.inject_reward
        feed_cmd = 0.0
        bite_cmd = 0.0
        near_food = False
        near_nest = nest_dist < 3.0
        nearest_food_idx = -1
        nearest_food_dist = float('inf')

        # Finde nächste Nahrungsquelle
        for i, fs in enumerate(self.food_sources):
            d = np.linalg.norm(pos - fs['pos'])
            if d < nearest_food_dist and self.food_amounts.get(i, 0) > 0:
                nearest_food_dist = d
                nearest_food_idx = i

        near_food = nearest_food_dist < 3.0

        # --- Futter aufheben ---
        if near_food and not self.carrying_food:
            self.food_pickup_timer += 1
            feed_cmd = 0.6   # Mandibeln arbeiten
            bite_cmd = 0.3   # Greifen
            reward = max(reward, 0.3)

            if self.food_pickup_timer > 40:  # ~20ms Aufhebezeit
                self.carrying_food = True
                self.carried_food_type = self.food_sources[nearest_food_idx]['type']
                self.food_amounts[nearest_food_idx] -= 1.0
                self.food_collected += 1
                self.food_pickup_timer = 0
                reward = 1.0  # Starke Belohnung beim Aufheben
                # Futter-Pheromon am Fundort hinterlassen
                fx = int(np.clip(self.food_sources[nearest_food_idx]['pos'][0], 0, 99))
                fy = int(np.clip(self.food_sources[nearest_food_idx]['pos'][1], 0, 99))
                for ddx in range(-2, 3):
                    for ddy in range(-2, 3):
                        if 0 <= fx+ddx < 100 and 0 <= fy+ddy < 100:
                            self.pheromone_food[fx+ddx, fy+ddy] = min(
                                self.pheromone_food[fx+ddx, fy+ddy] + 0.5, 2.0)
        else:
            self.food_pickup_timer = max(0, self.food_pickup_timer - 1)

        # --- Futter am Nest ablegen ---
        if near_nest and self.carrying_food:
            self.food_deposit_timer += 1
            feed_cmd = 0.4
            bite_cmd = 0.0   # Mandibeln öffnen

            if self.food_deposit_timer > 20:  # Schnelleres Ablegen
                self.carrying_food = False
                self.carried_food_type = None
                self.food_at_nest += 1
                self.food_deposit_timer = 0
                self.satiation = min(1.0, self.satiation + 0.1)
                self.energy = min(1.0, self.energy + 0.15)
                reward = 0.8  # Belohnung beim Ablegen
        else:
            self.food_deposit_timer = max(0, self.food_deposit_timer - 1)

        # === PHEROMON LEGEN ===
        pgx, pgy = int(np.clip(pos[0], 0, 99)), int(np.clip(pos[1], 0, 99))
        if self.carrying_food:
            # Beim Heimtragen: starke Spur-Pheromone (Rekrutierung)
            self.pheromone_trail[pgx, pgy] = min(
                self.pheromone_trail[pgx, pgy] + 0.08, 3.0)
        elif not near_nest:
            # Beim Erkunden: schwache Spur-Pheromone
            self.pheromone_trail[pgx, pgy] = min(
                self.pheromone_trail[pgx, pgy] + 0.005, 1.0)

        # Alarm-Pheromon bei inject
        if inject_type == 'alarm':
            self.pheromone_alarm[pgx, pgy] = min(
                self.pheromone_alarm[pgx, pgy] + 0.3, 2.0)

        # === GESCHWINDIGKEIT ===
        if near_food and not self.carrying_food:
            speed = 0.05  # Langsam beim Futtersuche
        elif self.carrying_food:
            speed = 0.35  # Mittel beim Tragen (beladen)
        elif near_nest:
            speed = 0.2   # Langsam am Nest
        else:
            speed = 0.5   # Normal beim Erkunden

        # Energieabhängige Geschwindigkeit
        speed *= (0.5 + 0.5 * self.energy)

        compass = heading + np.random.normal(0, 0.03)

        # === GEHIRN-SCHRITT ===
        sensory = SensoryInput(
            odor_vector=odor_input,
            visual_field=visual_input,
            polarization_angle=compass * 0.5,
            compass_angle=compass,
            speed=speed,
            angular_velocity=self.locomotion.angular_velocity,
            reward=reward,
            bite_command=bite_cmd,
            feed_command=feed_cmd,
        )
        output = self.brain.step(sensory)

        # === LOKOMOTION ===
        bilateral_diff = right_smell.sum() - left_smell.sum()
        chemotaxis = np.array([max(0, -bilateral_diff * 2.0),
                               max(0, bilateral_diff * 2.0)])

        # Verhaltensabhängige Steuerung
        if self.carrying_food:
            # Beim Heimtragen: stärker auf CX-Homing hören
            steering = output.steering * 0.7 + chemotaxis * 0.3
            # Tendenz Richtung Nest
            nest_angle = np.arctan2(self.nest_pos[1] - pos[1], self.nest_pos[0] - pos[0])
            angle_diff = nest_angle - heading
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            nest_pull = np.array([max(0, -angle_diff) * 1.5,
                                  max(0, angle_diff) * 1.5])
            steering = steering * 0.4 + nest_pull * 0.6
        else:
            # Beim Erkunden: stärker auf Chemotaxis hören
            steering = output.steering * 0.3 + chemotaxis * 0.7

        # Lévy-Walk (correlated random walk mit gelegentlichen Richtungswechseln)
        if not self.carrying_food:
            steering += np.abs(np.random.normal(0, 0.4, 2))
            self.exploration_timer += 1
            # Ameisen machen alle paar Sekunden starke Richtungswechsel
            if np.random.random() < 0.015:
                turn_dir = np.random.choice([0, 1])
                steering[turn_dir] += np.random.uniform(1.5, 3.5)
        else:
            # Beim Tragen weniger zufällig
            steering += np.abs(np.random.normal(0, 0.15, 2))

        # Wandvermeidung
        margin = 12.0
        wall_force = np.zeros(2)
        if pos[0] < margin:
            wall_force[0] += (margin - pos[0]) * 0.4
        if pos[0] > 100 - margin:
            wall_force[0] -= (pos[0] - (100 - margin)) * 0.4
        if pos[1] < margin:
            wall_force[1] += (margin - pos[1]) * 0.4
        if pos[1] > 100 - margin:
            wall_force[1] -= (pos[1] - (100 - margin)) * 0.4

        if np.linalg.norm(wall_force) > 0.1:
            desired_angle = np.arctan2(wall_force[1], wall_force[0])
            angle_diff = desired_angle - heading
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            wall_strength = np.linalg.norm(wall_force)
            if angle_diff > 0:
                steering[1] += wall_strength
            else:
                steering[0] += wall_strength

        steering = np.clip(steering, 0, 5.0)

        loco_result = self.locomotion.step(steering, speed, self.dt)
        self.leg_states = loco_result['leg_states'].tolist()

        # Arena-Grenzen
        self.locomotion.position = np.clip(self.locomotion.position, 1, 99)

        # Home-Vector zurücksetzen wenn Nest erreicht
        if near_nest:
            self.brain.central_complex.reset_path_integration()

        # -- Spike-Raten sammeln --
        al_spikes = self.brain.antennal_lobe.pn_output
        al_rate = al_spikes.sum() / max(len(al_spikes), 1)

        kc_spikes = output.kc_sparseness
        mbon_rate = output.learned_value.sum() / max(len(output.learned_value), 1)

        ol_out = self.brain.optic_lobe.visual_output
        ol_rate = ol_out.sum() / max(len(ol_out), 1)

        cx_bump = self.brain.central_complex.ring_attractor.bump
        cx_rate = cx_bump.sum() / max(len(cx_bump), 1)

        lh_act = self.brain.lateral_horn.channel_activity
        lh_rate = lh_act.sum() / max(len(lh_act), 1)

        seg_rate = (output.mandible_speed + output.mandible_force) / 2

        for key, val in [('AL', al_rate), ('MB_KC', kc_spikes),
                         ('MB_MBON', mbon_rate), ('OL', ol_rate),
                         ('CX', cx_rate), ('LH', lh_rate), ('SEG', seg_rate)]:
            self.region_spike_rates[key].append(float(val))
            if len(self.region_spike_rates[key]) > self.spike_window:
                self.region_spike_rates[key] = self.region_spike_rates[key][-self.spike_window:]

        # Position-Historie
        self.position_history.append(self.locomotion.position.copy().tolist())
        if len(self.position_history) > self.max_history:
            self.position_history = self.position_history[-self.max_history:]

        self.step_count += 1
        self.sim_time_ms += self.dt

        # Pheromon-Zusammenfassung für Dashboard (downsampled auf 20x20)
        trail_grid = []
        for ty in range(0, 100, 5):
            row = []
            for tx in range(0, 100, 5):
                val = float(self.pheromone_trail[tx:tx+5, ty:ty+5].max())
                row.append(round(val, 3) if val > 0.02 else 0)
            trail_grid.append(row)

        # State aktualisieren
        with self.lock:
            self.state = {
                'step': self.step_count,
                'time_ms': round(self.sim_time_ms, 1),
                'time_s': round(self.sim_time_ms / 1000, 2),
                'paused': self.paused,
                'speed': self.speed,

                # Gehirn-Info
                'total_neurons': self.brain.total_neurons,
                'neuron_counts': self.brain.neuron_counts,

                # Position & Navigation
                'position': self.locomotion.position.tolist(),
                'heading': round(float(self.locomotion.heading), 4),
                'heading_deg': round(float(np.degrees(self.locomotion.heading)), 1),
                'home_vector': output.home_vector.tolist(),
                'home_distance': round(float(output.home_distance), 2),
                'steering': output.steering.tolist(),
                'position_history': self.position_history[-500:],

                # Verhaltenszustand
                'behavioral_state': output.behavioral_state,
                'alarm_level': round(float(output.alarm_level), 4),
                'attraction_level': round(float(output.attraction_level), 4),

                # Lateralhorn-Kanäle
                'lh_channels': {
                    'alarm': round(float(lh_act[0]), 4),
                    'attraction': round(float(lh_act[1]), 4),
                    'avoidance': round(float(lh_act[2]), 4),
                    'social': round(float(lh_act[3]), 4),
                },

                # Pilzkörper
                'kc_sparseness': round(float(output.kc_sparseness), 6),
                'kc_active': int(output.kc_sparseness * self.brain.mushroom_body.n_kc),
                'kc_total': self.brain.mushroom_body.n_kc,
                'mbon_values': [round(float(v), 4) for v in output.learned_value],
                'stdp_mean_weight': round(
                    float(self.brain.mushroom_body.syn_kc_mbon.get_mean_weight()), 4),

                # Neuromodulation
                'neuromodulation': {
                    k: round(v, 4)
                    for k, v in self.brain.neuromodulator.levels.items()
                },

                # Zentralkomplex
                'cx_bump': [round(float(b), 4) for b in cx_bump],

                # Spike-Raten (letzte N Schritte)
                'spike_rates': {
                    k: v[-100:] for k, v in self.region_spike_rates.items()
                },

                # Motorik
                'mandible_speed': round(float(output.mandible_speed), 4),
                'mandible_force': round(float(output.mandible_force), 4),
                'feeding': output.feeding_active,
                'leg_states': self.leg_states,

                # Biologisches Verhalten
                'carrying_food': self.carrying_food,
                'carried_food_type': self.carried_food_type,
                'food_collected': self.food_collected,
                'food_at_nest': self.food_at_nest,
                'energy': round(self.energy, 4),
                'satiation': round(self.satiation, 4),

                # Umgebung
                'nest': self.nest_pos.tolist(),
                'food_sources': [
                    {'pos': fs['pos'].tolist(), 'type': fs['type'],
                     'amount': round(self.food_amounts.get(i, 0), 1)}
                    for i, fs in enumerate(self.food_sources)
                ],

                # Pheromon-Grid (20x20 downsampled)
                'pheromone_trail': trail_grid,

                # Injektionen
                'inject_odor': self.inject_odor,
                'inject_concentration': self.inject_concentration,
            }

    def get_state(self):
        with self.lock:
            return dict(self.state)


# Globale Simulation
sim = Simulation()


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP-Handler für Dashboard."""

    def do_GET(self):
        try:
            parsed = urlparse(self.path)

            if parsed.path == '/' or parsed.path == '/index.html':
                self._serve_html()
            elif parsed.path == '/api/state':
                self._serve_json(sim.get_state())
            elif parsed.path == '/api/stream':
                self._serve_sse()
            elif parsed.path == '/api/info':
                stats = sim.brain.get_statistics()
                stats['neuron_counts'] = {k: int(v) for k, v in stats['neuron_counts'].items()}
                self._serve_json(stats)
            elif parsed.path == '/favicon.ico':
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self.send_error(500, str(e))
            except Exception:
                pass

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length else b'{}'

            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                data = {}
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            return

        try:
            if parsed.path == '/api/control':
                action = data.get('action', '')
                if action == 'pause':
                    sim.paused = True
                elif action == 'resume':
                    sim.paused = False
                elif action == 'toggle':
                    sim.paused = not sim.paused
                elif action == 'speed':
                    sim.speed = float(data.get('value', 1.0))
                elif action == 'reset':
                    sim.paused = True
                    time.sleep(0.1)
                    sim._init_brain()
                    sim.step_count = 0
                    sim.sim_time_ms = 0
                    sim.region_spike_rates = {k: [] for k in sim.region_spike_rates}
                    sim.position_history = [sim.nest_pos.copy().tolist()]
                    sim.carrying_food = False
                    sim.carried_food_type = None
                    sim.food_collected = 0
                    sim.food_at_nest = 0
                    sim.satiation = 0.5
                    sim.energy = 1.0
                    sim.pheromone_trail[:] = 0
                    sim.pheromone_alarm[:] = 0
                    sim.pheromone_food[:] = 0
                    for i, fs in enumerate(sim.food_sources):
                        sim.food_amounts[i] = fs['amount']
                    sim.paused = False
                self._serve_json({'ok': True, 'paused': sim.paused})

            elif parsed.path == '/api/inject':
                sim.inject_odor = data.get('odor', None)
                sim.inject_concentration = float(data.get('concentration', 0.5))
                sim.inject_reward = float(data.get('reward', 0.0))
                self._serve_json({'ok': True})

            elif parsed.path == '/api/clear_inject':
                sim.inject_odor = None
                sim.inject_concentration = 0.0
                sim.inject_reward = 0.0
                self._serve_json({'ok': True})

            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self.send_error(500, str(e))
            except Exception:
                pass

    def _serve_json(self, data):
        body = json.dumps(data, default=_json_default).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self):
        """Server-Sent Events für Echtzeit-Updates."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        try:
            last_step = -1
            while True:
                state = sim.get_state()
                step = state.get('step', 0)
                if step != last_step:
                    data = json.dumps(state, default=_json_default)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                    last_step = step
                time.sleep(0.05)  # 20 Hz SSE updates
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception:
            pass

    def _serve_html(self):
        html_path = os.path.join(os.path.dirname(__file__), 'index.html')
        with open(html_path, 'rb') as f:
            content = f.read()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(content))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        # Nur Fehler loggen
        if '200' not in str(args):
            super().log_message(format, *args)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer mit Threading für gleichzeitige SSE + HTTP."""
    daemon_threads = True
    allow_reuse_address = True


def main():
    sim.start()
    server = ThreadedHTTPServer(('0.0.0.0', PORT), DashboardHandler)
    print(f"\n{'='*60}")
    print(f"  AMEISENGEHIRN DASHBOARD")
    print(f"  http://localhost:{PORT}")
    print(f"{'='*60}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sim.running = False
        server.shutdown()
        print("\nServer beendet.")


if __name__ == '__main__':
    main()
