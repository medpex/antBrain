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
            {'pos': np.array([25.0, 30.0]), 'type': 'sugar', 'conc': 0.8},
            {'pos': np.array([75.0, 70.0]), 'type': 'protein', 'conc': 0.6},
        ]
        self.trail_points = [
            np.array([48.0, 49.0]),
            np.array([45.0, 48.0]),
            np.array([42.0, 46.0]),
            np.array([40.0, 44.0]),
            np.array([37.0, 41.0]),
            np.array([35.0, 38.0]),
            np.array([32.0, 36.0]),
            np.array([30.0, 34.0]),
            np.array([27.0, 32.0]),
        ]

        # Position
        self.locomotion.position = self.nest_pos.copy()
        self.locomotion.heading = np.random.uniform(0, 2 * np.pi)

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
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            t0 = time.time()
            self._step()
            elapsed = time.time() - t0

            # Geschwindigkeitsregelung
            target_dt = 0.02 / max(self.speed, 0.1)
            sleep_time = max(0, target_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _step(self):
        pos = self.locomotion.position
        heading = self.locomotion.heading

        # -- Geruchsquellen --
        odor_sources = []
        for fs in self.food_sources:
            food_vec = np.zeros(400)
            food_vec[35:55] = 1.0
            odor_sources.append({
                'position': fs['pos'], 'concentration': fs['conc'],
                'vector': food_vec,
            })
        for tp in self.trail_points:
            trail_vec = np.zeros(400)
            trail_vec[5:15] = 1.0
            odor_sources.append({
                'position': tp, 'concentration': 0.3, 'vector': trail_vec,
            })

        # Injektionen
        inject_type = self.inject_odor
        inject_conc = self.inject_concentration
        if inject_type:
            inj_vec = np.zeros(400)
            if inject_type == 'alarm':
                inj_vec[0:5] = 1.0
            elif inject_type == 'food':
                inj_vec[35:55] = 1.0
            elif inject_type == 'trail':
                inj_vec[5:15] = 1.0
            elif inject_type == 'nestmate':
                inj_vec[15:35] = 1.0
            odor_sources.append({
                'position': pos, 'concentration': inject_conc,
                'vector': inj_vec,
            })

        # Antennen
        left_smell = self.left_antenna.sense(odor_sources, pos, heading, self.dt)
        right_smell = self.right_antenna.sense(odor_sources, pos, heading, self.dt)
        odor_input = (left_smell + right_smell) / 2.0

        # Augen
        landmarks = [
            {'angle': np.arctan2(self.nest_pos[1] - pos[1], self.nest_pos[0] - pos[0]),
             'distance': np.linalg.norm(pos - self.nest_pos),
             'size': 5.0, 'brightness': 0.8},
        ]
        for fs in self.food_sources:
            landmarks.append({
                'angle': np.arctan2(fs['pos'][1] - pos[1], fs['pos'][0] - pos[0]),
                'distance': np.linalg.norm(pos - fs['pos']),
                'size': 2.0, 'brightness': 0.5,
            })
        visual_input = self.left_eye.process_scene(landmarks=landmarks, heading=heading)

        # Belohnung
        reward = self.inject_reward
        for fs in self.food_sources:
            if np.linalg.norm(pos - fs['pos']) < 3.0:
                reward = max(reward, 0.5)

        compass = heading + np.random.normal(0, 0.03)

        # -- Gehirn-Schritt --
        sensory = SensoryInput(
            odor_vector=odor_input,
            visual_field=visual_input,
            polarization_angle=compass * 0.5,
            compass_angle=compass,
            speed=0.5,
            angular_velocity=self.locomotion.angular_velocity,
            reward=reward,
        )
        output = self.brain.step(sensory)

        # Lokomotion
        bilateral_diff = right_smell.sum() - left_smell.sum()
        chemotaxis = np.array([max(0, -bilateral_diff * 2.0),
                               max(0, bilateral_diff * 2.0)])
        steering = output.steering * 0.3 + chemotaxis * 0.7

        # Random-Walk mit gelegentlichen abrupten Richtungswechseln (Lévy Walk)
        steering += np.abs(np.random.normal(0, 0.5, 2))
        if np.random.random() < 0.02:  # ~2% Chance pro Step für starke Drehung
            turn_dir = np.random.choice([0, 1])
            steering[turn_dir] += np.random.uniform(2.0, 4.0)

        # Wandvermeidung — berechne Richtung weg von Wänden
        margin = 15.0
        wall_force = np.zeros(2)  # (fx, fy) weg von Wänden
        if pos[0] < margin:
            wall_force[0] += (margin - pos[0]) * 0.3
        if pos[0] > 100 - margin:
            wall_force[0] -= (pos[0] - (100 - margin)) * 0.3
        if pos[1] < margin:
            wall_force[1] += (margin - pos[1]) * 0.3
        if pos[1] > 100 - margin:
            wall_force[1] -= (pos[1] - (100 - margin)) * 0.3

        if np.linalg.norm(wall_force) > 0.1:
            # Gewünschte Richtung weg von der Wand
            desired_angle = np.arctan2(wall_force[1], wall_force[0])
            angle_diff = desired_angle - heading
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            wall_strength = np.linalg.norm(wall_force)
            if angle_diff > 0:
                steering[1] += wall_strength
            else:
                steering[0] += wall_strength

        steering = np.clip(steering, 0, 5.0)

        self.locomotion.step(steering, 0.5, self.dt)

        # Arena-Grenzen
        self.locomotion.position = np.clip(self.locomotion.position, 1, 99)

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

                # Umgebung
                'nest': self.nest_pos.tolist(),
                'food_sources': [
                    {'pos': fs['pos'].tolist(), 'type': fs['type']}
                    for fs in self.food_sources
                ],

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
        parsed = urlparse(self.path)

        if parsed.path == '/' or parsed.path == '/index.html':
            self._serve_html()
        elif parsed.path == '/api/state':
            self._serve_json(sim.get_state())
        elif parsed.path == '/api/info':
            stats = sim.brain.get_statistics()
            stats['neuron_counts'] = {k: int(v) for k, v in stats['neuron_counts'].items()}
            self._serve_json(stats)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length else b'{}'

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

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

    def _serve_json(self, data):
        body = json.dumps(data, default=_json_default).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

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


def main():
    sim.start()
    server = HTTPServer(('0.0.0.0', PORT), DashboardHandler)
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
