import pandas as pd
import numpy as np
from datetime import datetime

Raw = 'adaptor.cams_solar_rad2.retrieve-1641471401.5858812-3176-8-672a034a-2ec7-445a-8eec-04608061f019.csv'

class Copernicus:
    def __init__(self, dir):
        self.dir = dir
        self.raw = pd.read_csv(dir, delimiter=';')

    def convert_time(self):
        self.ObservationPeriod = self.raw['Observation period'].to_numpy()
        self.ObservationPeriod = np.asarray([datetime.strptime(i.split(sep='/')[0],"%Y-%m-%dT%H:%M:%S.0") for i in self.ObservationPeriod])

    def irradiation_components(self):
        self.TOA = self.raw['TOA'].to_numpy(dtype=float)
        self.ClearSky_GHI = self.raw['Clear sky GHI'].to_numpy(dtype=float)
        self.ClearSky_BHI = self.raw['Clear sky BHI'].to_numpy(dtype=float)
        self.ClearSky_DHI = self.raw['Clear sky DHI'].to_numpy(dtype=float)
        self.ClearSky_BNI = self.raw['Clear sky BNI'].to_numpy(dtype=float)
        self.GHI = self.raw['GHI'].to_numpy(dtype=float)
        self.BHI = self.raw['BHI'].to_numpy(dtype=float)
        self.DHI = self.raw['DHI'].to_numpy(dtype=float)
        self.BNI = self.raw['BNI'].to_numpy(dtype=float)
        self.Reliability = self.raw['Reliability'].to_numpy(dtype=float)


Data = Copernicus(Raw)
Data.convert_time()
Data.irradiation_components()
