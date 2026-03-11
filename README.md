# antBrain — Ant Brain Simulator

A biologically realistic spiking neural network that simulates a complete ant brain with **~70,000 neurons** across 6 brain regions. Built with pure Python and NumPy.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Neurons](https://img.shields.io/badge/Neurons-70%2C919-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## What is this?

This project models the brain of an ant — from sensory input (smell, vision, taste) through neural processing to motor output (walking, biting, feeding). Every neuron fires individually using biophysical models. The ant perceives odors, navigates toward food, learns from rewards, and responds to pheromones — all emerging from the neural dynamics, not hardcoded behavior.

### Brain Regions

```
SENSORY INPUT
├── Antennae (Odor)  ──→  Antennal Lobe (400 glomeruli)
│                              ├──→ Mushroom Body (learning)
│                              └──→ Lateral Horn (instinct)
├── Eyes (Visual)    ──→  Optic Lobe (600 ommatidia)
│                              └──→ Central Complex (compass)
└── Gustatory        ──→  SEG (taste + mandibles)

INTEGRATION
├── Mushroom Body    50,000 Kenyon Cells, sparse coding, STDP learning
├── Lateral Horn     Innate responses: alarm, attraction, avoidance, social
└── Central Complex  Ring attractor compass, path integration, steering

MOTOR OUTPUT
└── Central Complex → Locomotion (CPG tripod gait)
```

### Key Features

| Feature | Implementation |
|---|---|
| **Neuron Models** | Leaky Integrate-and-Fire (large populations), Izhikevich (small functional groups) |
| **Learning** | STDP with dopamine modulation — the ant learns odor-reward associations in real time |
| **Sparse Coding** | ~2–5% Kenyon Cell activation, similar to real insect brains |
| **Navigation** | Ring attractor for head direction, path integration for home vector |
| **Innate Behavior** | Hardwired alarm/attraction circuits in Lateral Horn |
| **Neuromodulation** | Dopamine, Octopamine, Serotonin, Tyramine — modulate learning and behavior |
| **Efficient** | scipy.sparse matrices for large synaptic connections |

## Setup

```bash
# Clone
git clone https://github.com/medpex/antBrain.git
cd antBrain

# Install dependencies
pip install numpy scipy matplotlib pyyaml

# Run tests
python -m pytest tests/

# Start the live dashboard
python dashboard/server.py
```

Open **http://localhost:8050** in your browser.

## Live Dashboard

The dashboard shows the ant's brain activity in real time:

- **Brain Map** — firing rates across all 6 regions
- **Arena** — ant position, food sources, trail pheromones, movement trail
- **Sidebar** — behavioral state, LH channel activity, neuromodulation levels, compass heading, Mushroom Body stats
- **Spike Rate Chart** — rolling neural activity per region
- **Pheromone Injection** — inject alarm, food, or trail pheromones and watch the brain react

### Controls

| Button | Effect |
|---|---|
| Pause / Resume | Stop or continue the simulation |
| Speed slider | 0.1x – 5x simulation speed |
| Reset | Restart with a fresh brain |
| Inject Alarm | Simulate alarm pheromone — triggers escape behavior |
| Inject Food | Simulate food odor — triggers foraging |
| Inject Trail | Simulate trail pheromone — triggers path following |

## Project Structure

```
antBrain/
├── ant_brain/
│   ├── core/
│   │   ├── neuron.py          # LIF & Izhikevich neuron models
│   │   ├── synapse.py         # Static synapses & STDP learning
│   │   └── brain.py           # Central brain connecting all regions
│   ├── regions/
│   │   ├── antennal_lobe.py   # Olfactory processing (400 glomeruli)
│   │   ├── mushroom_body.py   # Learning & memory (50k Kenyon cells)
│   │   ├── central_complex.py # Navigation (ring attractor + path integration)
│   │   ├── lateral_horn.py    # Innate odor responses
│   │   ├── optic_lobe.py      # Visual processing
│   │   └── subesophageal_ganglion.py  # Motor control
│   ├── sensors/
│   │   ├── antenna.py         # Olfactory receptor simulation
│   │   └── eye.py             # Compound eye with polarization
│   └── actuators/
│       └── locomotion.py      # CPG-based tripod gait
├── dashboard/
│   ├── server.py              # HTTP server + simulation loop
│   └── index.html             # Real-time visualization
├── tests/
│   └── test_brain.py          # Unit tests for all components
├── configs/
│   └── default.yaml           # Default brain configuration
└── simulations/
    └── demo_foraging.py       # Standalone foraging demo
```

## How does the ant behave?

The ant starts at its nest and explores the arena. Behavior emerges from neural activity:

1. **Trail detection** — antenna picks up trail pheromone, Lateral Horn attraction channel activates → `foraging` state
2. **Food approach** — bilateral antenna comparison drives chemotaxis steering toward food source
3. **Learning** — when near food, reward signal triggers dopamine release → STDP strengthens KC→MBON connections → ant associates this odor pattern with reward
4. **Homing** — Central Complex tracks outbound path via path integration → when carrying food, steering reverses toward nest
5. **Alarm response** — alarm pheromone injection activates alarm channel in Lateral Horn → overrides foraging → triggers avoidance

## API

The dashboard server exposes a simple API:

```
GET  /api/state          # Full brain state (JSON)
GET  /api/info           # Brain statistics
POST /api/control        # {"action": "pause|resume|reset|speed", "value": 2.0}
POST /api/inject         # {"odor": "alarm|food|trail", "concentration": 0.8}
POST /api/clear_inject   # Clear active injection
```

## References

- Zube et al. (2008) — Organization of the olfactory pathway in the ant brain
- Webb & Wystrach (2016) — Neural mechanisms of insect navigation
- Aso et al. (2014) — Mushroom body output neurons and dopaminergic neurons
- Seelig & Jayaraman (2015) — Ring attractor dynamics in the Drosophila central complex
- Hölldobler & Wilson (1990) — *The Ants*

---

Built with NumPy, scipy, and curiosity about how 250,000 neurons create intelligent behavior.
