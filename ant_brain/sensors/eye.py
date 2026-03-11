"""
Facettenauge - Visueller Input.

Ameisen haben 2 Facettenaugen mit je ~100-1300 Ommatidien
(artabhängig). Jedes Ommatidium erfasst einen kleinen
Ausschnitt des Gesichtsfelds.

Funktionen:
- Landmark-Erkennung für Navigation
- Polarisationslicht-Detektion (Dorsal Rim Area)
- Bewegungsdetektion
"""

import numpy as np


class CompoundEye:
    """
    Simuliertes Facettenauge der Ameise.

    Wandelt eine 2D-Szene in Ommatidien-Aktivierungen um.
    """

    def __init__(self, n_ommatidia: int = 600, fov_degrees: float = 180.0,
                 side: str = "left"):
        """
        Args:
            n_ommatidia: Anzahl Ommatidien
            fov_degrees: Gesichtsfeld in Grad
            side: 'left' oder 'right'
        """
        self.n_ommatidia = n_ommatidia
        self.fov = np.radians(fov_degrees)
        self.side = side

        # Ommatidien-Anordnung (halbkugelförmig)
        self.omm_angles = np.linspace(-self.fov/2, self.fov/2, n_ommatidia)

        # DRA (Dorsal Rim Area) für Polarisationslicht: obere ~8%
        self.n_dra = int(n_ommatidia * 0.08)

        # Akzeptanzwinkel jedes Ommatidiums (~2-5 Grad)
        self.acceptance_angle = np.radians(3.0)

    def process_scene(self, scene_intensities: np.ndarray = None,
                      landmarks: list[dict] = None,
                      heading: float = 0.0) -> np.ndarray:
        """
        Szenenverarbeitung.

        Args:
            scene_intensities: Direktes Intensitätsarray (n_ommatidia,)
            landmarks: Liste von Landmarks [{'angle': rad, 'distance': m, 'size': float}]
            heading: Aktuelle Blickrichtung (rad)

        Returns:
            Aktivierungsarray (n_ommatidia,) normalisiert 0-1
        """
        if scene_intensities is not None:
            return np.clip(scene_intensities, 0, 1)

        activation = np.zeros(self.n_ommatidia) + 0.05  # Hintergrundlicht

        if landmarks:
            for lm in landmarks:
                lm_angle = lm['angle'] - heading
                lm_distance = lm.get('distance', 10.0)
                lm_size = lm.get('size', 1.0)

                # Lateraler Offset für Seitenauswahl
                if self.side == 'left':
                    lm_angle -= 0.05
                else:
                    lm_angle += 0.05

                # Visuelle Größe abhängig von Distanz
                angular_size = lm_size / (lm_distance + 0.1)

                # Aktivierung der betroffenen Ommatidien
                angle_diffs = np.abs(self.omm_angles - lm_angle)
                affected = angle_diffs < (angular_size + self.acceptance_angle)
                activation[affected] += lm.get('brightness', 0.5) / (1 + lm_distance * 0.1)

        return np.clip(activation, 0, 1)

    def detect_polarization(self, sun_azimuth: float, heading: float) -> float:
        """
        Polarisationslicht-Detektion über DRA.

        Args:
            sun_azimuth: Azimutwinkel der Sonne (rad)
            heading: Aktuelle Blickrichtung (rad)

        Returns:
            Geschätzter Polarisationswinkel
        """
        # Polarisationsmuster ist senkrecht zur Sonneneinfallsrichtung
        pol_angle = sun_azimuth + np.pi / 2
        return pol_angle

    def reset(self):
        pass
