"""
AntBrain - Biologisch inspirierte Simulation eines Ameisengehirns
=================================================================

Modulare Spiking Neural Network Simulation basierend auf der
Neuroanatomie von Ameisen (~250.000 Neuronen).

Hirnregionen:
- AntennalLobe: Olfaktorische Verarbeitung (Pheromone)
- MushroomBody: Lernen, Gedächtnis, sensorische Integration
- OpticLobe: Visuelle Verarbeitung
- CentralComplex: Navigation, Motorkoordination
- LateralHorn: Angeborene olfaktorische Reaktionen
- SubesophagealGanglion: Mundwerkzeug-Steuerung
- LateralAccessoryLobe: Prämotorische Ausgabe
"""

from ant_brain.core.brain import AntBrain
from ant_brain.core.neuron import LIFNeuron, IzhikevichNeuron
from ant_brain.core.synapse import Synapse, STDPSynapse

__version__ = "0.1.0"
