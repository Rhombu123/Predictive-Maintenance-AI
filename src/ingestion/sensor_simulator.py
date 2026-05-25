import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class SensorSimulator:
    def __init__(self, equipment_id: str, seed: int = 42):
        self.equipment_id = equipment_id
        np.random.seed(seed)

    def generate(self, n_hours: int = 720) -> pd.DataFrame:
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(n_hours, 0, -1)]

        # Inject gradual drift to simulate a bearing wearing out
        drift = np.linspace(0, 5, n_hours)
        vibration = 2.5 + drift + np.random.normal(0, 0.3, n_hours)
        temperature = 70 + drift * 0.8 + np.random.normal(0, 1.2, n_hours)
        pressure = 100 - drift * 0.5 + np.random.normal(0, 0.5, n_hours)

        # Inject a few spike anomalies
        for i in np.random.choice(n_hours, size=10, replace=False):
            vibration[i] += np.random.uniform(4, 8)

        return pd.DataFrame({
            "timestamp": timestamps,
            "equipment_id": self.equipment_id,
            "vibration": vibration,
            "temperature": temperature,
            "pressure": pressure,
        })
