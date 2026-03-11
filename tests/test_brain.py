"""Tests für das Ameisengehirn."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ant_brain.core.neuron import LIFNeuron, IzhikevichNeuron, NeuronParams
from ant_brain.core.synapse import Synapse, STDPSynapse
from ant_brain.core.brain import AntBrain, SensoryInput


def test_lif_neuron():
    """Test LIF-Neuronenmodell."""
    neuron = LIFNeuron(100)

    # Kein Input → kein Spike
    spikes = neuron.step(np.zeros(100))
    assert not np.any(spikes), "Sollte ohne Input nicht feuern"

    # Starker Input → Spikes
    neuron.reset()
    any_spiked = False
    for _ in range(200):
        spikes = neuron.step(np.full(100, 5.0))
        if np.any(spikes):
            any_spiked = True
    assert any_spiked, "Sollte bei starkem Input feuern"

    # Neuromodulation
    neuron.set_modulation('dopamine', 1.5)
    assert neuron.modulation['dopamine'] == 1.5

    print("  LIF-Neuron: OK")


def test_izhikevich_neuron():
    """Test Izhikevich-Neuronenmodell."""
    neuron = IzhikevichNeuron(50, IzhikevichNeuron.REGULAR_SPIKING)

    # Starker Input → Spikes
    for _ in range(100):
        spikes = neuron.step(np.full(50, 15.0))

    total_spikes = neuron.spike_count.sum()
    assert total_spikes > 0, "Izhikevich-Neuron sollte feuern"

    # Bursting
    burst_neuron = IzhikevichNeuron(10, IzhikevichNeuron.BURSTING)
    for _ in range(100):
        burst_neuron.step(np.full(10, 15.0))

    print("  Izhikevich-Neuron: OK")


def test_synapse():
    """Test synaptische Übertragung."""
    syn = Synapse(50, 30, connectivity=0.5, weight_mean=0.5)

    pre_spikes = np.zeros(50, dtype=bool)
    pre_spikes[:10] = True

    # Erste Übertragung (Verzögerung)
    current = syn.transmit(pre_spikes)
    assert current.shape == (30,), f"Falsche Form: {current.shape}"

    # Nach Verzögerung sollte Strom ankommen
    for _ in range(20):
        current = syn.transmit(pre_spikes)
    assert np.any(current != 0), "Synaptischer Strom sollte nicht null sein"

    print("  Synapse: OK")


def test_stdp_synapse():
    """Test STDP-Lernen."""
    syn = STDPSynapse(100, 10, connectivity=0.3)

    mean_before = syn.get_mean_weight()

    # Pre- und Post-Spikes erzeugen
    for _ in range(100):
        pre = np.random.random(100) > 0.9
        post = np.random.random(10) > 0.9
        syn.set_dopamine(1.5)  # Belohnung
        syn.update_stdp(pre, post, dt=0.5)
        syn.transmit(pre)

    mean_after = syn.get_mean_weight()

    # Gewichte sollten sich geändert haben
    assert abs(mean_after - mean_before) > 1e-6, "STDP sollte Gewichte ändern"

    print("  STDP-Synapse: OK")


def test_brain_creation():
    """Test Gehirn-Instanziierung."""
    config = {
        'n_glomeruli': 20,
        'n_kenyon_cells': 1000,
        'n_ommatidia': 50,
        'cx_columns': 8,
        'lh_neurons': 200,
    }
    brain = AntBrain(config)

    assert brain.total_neurons > 0, "Sollte Neuronen haben"
    print(f"  Gehirn erstellt: {brain.total_neurons:,} Neuronen")
    print(f"  Regionen: {list(brain.neuron_counts.keys())}")
    print("  Brain-Erstellung: OK")


def test_brain_step():
    """Test Gehirn-Simulationsschritt."""
    config = {
        'n_glomeruli': 10,
        'n_kenyon_cells': 500,
        'n_ommatidia': 20,
        'cx_columns': 8,
        'lh_neurons': 100,
    }
    brain = AntBrain(config)

    sensory = SensoryInput(
        odor_type='food',
        odor_concentration=0.5,
        compass_angle=1.0,
        speed=0.3,
    )

    output = brain.step(sensory)

    assert output.steering is not None
    assert output.behavioral_state in ['idle', 'foraging', 'alarm', 'homing', 'feeding']
    assert 0 <= output.kc_sparseness <= 1

    print(f"  Brain-Step: OK (Zustand: {output.behavioral_state})")


def test_brain_simulation():
    """Test mehrstufige Simulation."""
    config = {
        'n_glomeruli': 10,
        'n_kenyon_cells': 500,
        'n_ommatidia': 20,
        'cx_columns': 8,
        'lh_neurons': 100,
    }
    brain = AntBrain(config)

    # 100 Schritte mit Nahrungsgeruch
    for i in range(100):
        reward = 1.0 if i == 50 else 0.0
        sensory = SensoryInput(
            odor_type='food',
            odor_concentration=0.5,
            compass_angle=float(i) * 0.01,
            speed=0.3,
            reward=reward,
        )
        output = brain.step(sensory)

    stats = brain.get_statistics()
    assert stats['steps'] == 100
    assert stats['time_ms'] > 0
    print(f"  Simulation: OK ({stats['steps']} Schritte, {stats['time_ms']:.1f}ms)")


def run_all_tests():
    """Alle Tests ausführen."""
    print("=" * 50)
    print("AMEISENGEHIRN - TESTS")
    print("=" * 50)

    tests = [
        test_lif_neuron,
        test_izhikevich_neuron,
        test_synapse,
        test_stdp_synapse,
        test_brain_creation,
        test_brain_step,
        test_brain_simulation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FEHLER in {test.__name__}: {e}")
            failed += 1

    print()
    print(f"Ergebnis: {passed} bestanden, {failed} fehlgeschlagen")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
