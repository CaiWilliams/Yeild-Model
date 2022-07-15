import numpy as np
import pandas as pd
import xarray as xr
import itertools
#import cfgrib
from datetime import datetime
#import calendar
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#import copy
import os


class Copernicus:

    def __init__(self, dir):
        self.dir = dir
        self.raw = pd.read_csv(dir, delimiter=';')

    def convert_time(self):
        self.ObservationPeriod = self.raw['Observation period'].to_numpy()
        start = datetime.strptime(self.ObservationPeriod[0].split(sep='/')[0], "%Y-%m-%dT%H:%M:%S.0")
        end = datetime.strptime(self.ObservationPeriod[0].split(sep='/')[1], "%Y-%m-%dT%H:%M:%S.0")
        diff = end - start
        self.ObservationPeriod = np.asarray([datetime.strptime(i.split(sep='/')[0], "%Y-%m-%dT%H:%M:%S.0") + diff/2 for i in self.ObservationPeriod])
        return

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
        return


class Farm:
    def __init__(self, latitude, longitude, elevation, albedo, weather_dir=None):
        self.Latitude_deg = latitude
        self.Latitude_rad = np.deg2rad(self.Latitude_deg)
        self.Longitude_deg = longitude
        self.Longitude_rad = np.deg2rad(self.Longitude_deg)
        self.albedo = albedo
        self.Elevation = elevation
        self.weather_dir = weather_dir

    def load_weather(self):
        grid_data = xr.load_dataset(self.weather_dir, engine='cfgrib')

        Temperature = grid_data['t2m']
        Temperature = Temperature[dict(latitude=int(np.round(self.Latitude_deg, 1)), longitude=int(np.round(self.Longitude_deg, 1)))]
        self.Temperature = np.array(Temperature).flatten() - 273.15

        u10 = grid_data['u10']
        u10 = u10[dict(latitude=int(np.round(self.Latitude_deg, 1)), longitude=int(np.round(self.Longitude_deg, 1)))]
        self.West_East_Wind = np.array(u10).flatten()

        v10 = grid_data['v10']
        v10 = v10[dict(latitude=int(np.round(self.Latitude_deg, 1)), longitude=int(np.round(self.Longitude_deg, 1)))]
        self.South_North_Wind = np.array(v10).flatten()

    def save_weather(self):
        grid_data = xr.load_dataset(self.weather_dir, engine='cfgrib')

        Temperature = grid_data['t2m']
        Temperature = Temperature[dict(latitude=int(np.round(self.Latitude_deg, 1)), longitude=int(np.round(self.Longitude_deg, 1)))]
        self.Temperature = np.array(Temperature).flatten() - 273.15

        u10 = grid_data['u10']
        u10 = u10[dict(latitude=int(np.round(self.Latitude_deg, 1)), longitude=int(np.round(self.Longitude_deg, 1)))]
        self.West_East_Wind = np.array(u10).flatten()

        v10 = grid_data['v10']
        v10 = v10[dict(latitude=int(np.round(self.Latitude_deg, 1)), longitude=int(np.round(self.Longitude_deg, 1)))]
        self.South_North_Wind = np.array(v10).flatten()

        weathercsv = pd.DataFrame()
        weathercsv['Temperature'] = self.Temperature
        weathercsv['West_East_Wind'] = self.West_East_Wind
        weathercsv['South_North_Wind'] = self.South_North_Wind
        weathercsv.dropna(axis=0,how='any',inplace=True)
        weathercsv.to_csv('Weather/Weatherdata.csv',index=False)

    def load_weather_csv(self, weather_csv_dir):
        weathercsv = pd.read_csv(weather_csv_dir)
        self.Temperature = weathercsv['Temperature'].to_numpy()
        self.West_East_Wind = weathercsv['West_East_Wind'].to_numpy()
        self.South_North_Wind = weathercsv['South_North_Wind'].to_numpy()

    def load_weather_csv_2(self, weather_csv_dir):
        weathercsv = pd.read_csv(weather_csv_dir)
        self.Temperature = weathercsv['2t'].to_numpy()
        self.West_East_Wind = weathercsv['10u'].to_numpy()
        self.South_North_Wind = weathercsv['10v'].to_numpy()


class Sun:
    def __init__(self, farm, data):
        self.farm = farm
        self.observation_period = data.ObservationPeriod
        self.TOA = data.TOA
        self.ClearSky_GHI = data.ClearSky_GHI
        self.ClearSky_BHI = data.ClearSky_BHI
        self.ClearSky_DHI = data.ClearSky_DHI
        self.ClearSky_BNI = data.ClearSky_BNI
        self.GHI = data.GHI
        self.BHI = data.BHI
        self.DHI = data.DHI
        self.BNI = data.BNI

    def timing(self):
        self.Hour = np.asarray([i.hour for i in self.observation_period])
        self.Day = np.asarray([i.timetuple().tm_yday + (i.hour/24) for i in self.observation_period])
        self.Minute = np.asarray([i.minute for i in self.observation_period])
        self.Second = np.asarray((i.second for i in self.observation_period))
        self.JDfrac = (self.Hour/24 + self.Minute/(24*60)) - 0.5
        self.HourAngle_deg = 15 * (self.Hour - 12)
        self.HourAngle_rad = np.deg2rad(self.HourAngle_deg)
        self.Year = np.asarray([i.year for i in self.observation_period])
        self.Month = np.asarray([i.month for i in self.observation_period])
        self.julian_day()
        return

    def julian_day(self):
        self.JulianDay = np.asarray([i.toordinal() + 1721425 for i in self.observation_period])
        self.JulianDay = self.JulianDay + self.JDfrac
        t = (self.Year - 2000)/100
        self.DeltaT = 102 + (102*t) + (25.3*np.power(t, 2))
        self.DeltaT = self.DeltaT + (0.37 * (self.Year - 2100))
        self.DeltaT = 0
        self.JulianEphemerisDay = self.JulianDay + self.DeltaT/86400
        self.JulianCentury = (self.JulianDay - 2451545)/36525
        self.JulianEphemerisCentury = (self.JulianEphemerisDay - 2451545)/36525
        self.JulianEphemerisMillennium = (self.JulianEphemerisCentury/10)

    def load_earth_periodic_terms(self):
        dir = 'EarthPeriodicTerms'
        files = os.listdir(dir)
        filesdirs = [ dir + "\\" + i for i in files]
        filenames = [i.split('.')[0] for i in files]
        self.EPT = dict()

        for idx, file in enumerate(filesdirs):
            data = np.genfromtxt(file, delimiter=',', skip_header=1)
            try:
                A = data[:, 0]
                B = data[:, 1]
                C = data[:, 2]
            except:
                A = data[0]
                B = data[1]
                C = data[2]
            temps = dict()
            temps['A'] = A
            temps['B'] = B
            temps['C'] = C
            self.EPT[filenames[idx]] = temps

    def heliocentric_properties(self):
        LongitudeTerms = ['L0','L1','L2','L3','L4','L5']
        LatitudeTerms = ['B0','B1']
        RadiusTerms = ['R0','R1','R2','R3','R4']
        self.load_earth_periodic_terms()
        LSums = self.Calc_helioc_terms(LongitudeTerms)
        BSums = self.Calc_helioc_terms(LatitudeTerms)
        RSums = self.Calc_helioc_terms(RadiusTerms)

        self.L_rad = (LSums[0] + LSums[1] * self.JulianEphemerisMillennium + LSums[2] * np.power(self.JulianEphemerisMillennium, 2) + LSums[3] * np.power(self.JulianEphemerisMillennium, 3) + LSums[4] * np.power(self.JulianEphemerisMillennium, 4) + LSums[5] * np.power(self.JulianEphemerisMillennium, 5))/np.power(10, 8)
        self.L_deg = np.rad2deg(self.L_rad)

        self.B_rad = (BSums[0] + (BSums[1] * self.JulianEphemerisMillennium))/np.power(10, 8)
        self.B_deg = np.rad2deg(self.B_rad)

        self.R_AU = (RSums[0] + RSums[1] * self.JulianEphemerisMillennium + RSums[2] * np.power(self.JulianEphemerisMillennium, 2) + RSums[3] * np.power(self.JulianEphemerisMillennium, 3) + RSums[4] * np.power(self.JulianEphemerisMillennium, 4))/np.power(10, 8)
        return

    def Calc_helioc_terms(self,terms):
        Results = np.zeros([len(terms),len(self.JulianEphemerisMillennium)])
        A = np.asarray([self.EPT[i]['A'] for i in terms], dtype=object)
        B = np.asarray([self.EPT[i]['B'] for i in terms], dtype=object)
        C = np.array([self.EPT[i]['C'] for i in terms], dtype=object)
        for idx, i in enumerate(self.JulianEphemerisMillennium):
            CosTerm = B * C * i
        for jdx,j in enumerate(terms):
            Results[jdx] = np.sum(A[jdx] * np.cos(np.deg2rad(CosTerm[jdx])))
        return Results

    def geocentric_lat_long(self):
        self.Geo_Longitude_deg = self.L_deg + 180
        self.Geo_Longitude_rad = np.deg2rad(self.Geo_Longitude_deg)
        self.Geo_Latitude = -self.B_deg
        return

    def nutatiion_in_longitude_and_obliquity(self):
        X_0 = 297.85036 + 445267.111480 * self.JulianEphemerisCentury - 0.0019142 * np.power(self.JulianEphemerisCentury, 2) + np.power(self.JulianEphemerisCentury, 3)/189474
        X_1 = 357.52772 + 35999.050340 * self.JulianEphemerisCentury - 0.0001603 * np.power(self.JulianEphemerisCentury, 2) - np.power(self.JulianEphemerisCentury, 3)/300000
        X_2 = 134.96298 + 477198.867398 * self.JulianEphemerisCentury + 0.008672 * np.power(self.JulianEphemerisCentury, 2) + np.power(self.JulianEphemerisCentury, 3)/56250
        X_3 = 93.27191 + 48202.017538 * self.JulianEphemerisCentury - 0.0036825 * np.power(self.JulianEphemerisCentury, 2) + np.power(self.JulianEphemerisCentury, 3)/328270
        X_4 = 125.04452 - 1934.126261 * self.JulianEphemerisCentury + 0.0020708 * np.power(self.JulianEphemerisCentury, 2) + np.power(self.JulianEphemerisCentury, 3)/450000
        X = [X_0, X_1, X_2, X_3, X_4]
        self.load_nutation_coeffs()
        self.calc_nutation_longitude(X)
        self.calc_nutation_obliquity(X)

    def load_nutation_coeffs(self):
        dir = 'Nutation'
        file = 'coefficients.csv'
        data = np.genfromtxt(dir + '\\' + file, delimiter=',', skip_header=1)
        data = np.nan_to_num(data,0)
        self.NutationCoeffs = dict()
        self.NutationCoeffs['Y0'] = data[:,0]
        self.NutationCoeffs['Y1'] = data[:,1]
        self.NutationCoeffs['Y2'] = data[:,2]
        self.NutationCoeffs['Y3'] = data[:,3]
        self.NutationCoeffs['Y4'] = data[:,4]
        self.NutationCoeffs['a'] = data[:,5]
        self.NutationCoeffs['b'] = data[:,6]
        self.NutationCoeffs['c'] = data[:,7]
        self.NutationCoeffs['d'] = data[:,8]
        self.Nu_Y0 = np.nan_to_num(self.NutationCoeffs['Y0'])
        self.Nu_Y1 = np.nan_to_num(self.NutationCoeffs['Y1'])
        self.Nu_Y2 = np.nan_to_num(self.NutationCoeffs['Y2'])
        self.Nu_Y3 = np.nan_to_num(self.NutationCoeffs['Y3'])
        self.Nu_Y4 = np.nan_to_num(self.NutationCoeffs['Y4'])
        self.Nu_a = np.nan_to_num(self.NutationCoeffs['a'])
        self.Nu_b = np.nan_to_num(self.NutationCoeffs['b'])
        self.Nu_c = np.nan_to_num(self.NutationCoeffs['c'])
        self.Nu_d = np.nan_to_num(self.NutationCoeffs['d'])

    def calc_nutation_longitude(self,X):
        Results = np.zeros((len(self.JulianEphemerisCentury)))
        for idx, JCE in enumerate(range(len(self.JulianEphemerisCentury))):
            Results[idx] = np.sum(self.Nu_a[:] + self.Nu_b[:] * JCE * np.sin((np.sum([(X[0][idx] * self.Nu_Y0[:]), (X[1][idx] * self.Nu_Y1[:]), (X[2][idx] * self.Nu_Y2[:]), (X[3][idx] * self.Nu_Y3[:]), (X[4][idx] * self.Nu_Y4[:])]))),axis=0)/36000000

        #for idx in range(len(self.JulianEphemerisCentury)):
        self.Longitude_Nu = Results#np.sum(Results,axis=0)/36000000
        #self.DeltaNutationLon = np.zeros(50)
        #for i in range(len(self.JulianEphemerisCentury[0:50])):
        #    self.DeltaNutationLon[i] = np.sum((self.NutationCoeffs['a'] + self.NutationCoeffs['b'] * self.JulianEphemerisCentury) * np.sin((X[0]*self.NutationCoeffs['Y0'])+(X[1]*self.NutationCoeffs['Y1'])+(X[2]*self.NutationCoeffs['Y2'][i])+(X[3]*self.NutationCoeffs['Y3'])+(X[4]*self.NutationCoeffs['Y4'])))/36000000

    def calc_nutation_obliquity(self,X):
        Results = np.zeros(len(self.JulianEphemerisCentury))
        for idx, JCE in enumerate(range(len(self.JulianEphemerisCentury))):
            Results[idx] = np.sum(self.Nu_c[:] + self.Nu_d[:] * JCE * np.cos(np.deg2rad(np.sum([(X[0][idx] * self.Nu_Y0[:]), (X[1][idx] * self.Nu_Y1[:]), (X[2][idx] * self.Nu_Y2[:]),(X[3][idx] * self.Nu_Y3[:]), (X[4][idx] * self.Nu_Y4[:])]))),axis=0)/36000000

        #for idx in range(len(self.JulianEphemerisCentury)):
        self.Obliquity_Nu = Results

    def true_obliquity(self):
        U = self.JulianEphemerisMillennium/10
        self.Mean_Obliquity = 84381.448 - 4680.93 * U - 1.55 * np.power(U, 2) + 1999.25 * np.power(U, 3) - 51.38 * np.power(U, 4) - 249.67 * np.power(U, 5) - 39.05 * np.power(U, 6) + 7.12 * np.power(U, 7) + 27.87 * np.power(U, 8) + 5.79 * np.power(U, 9) + 2.45 * np.power(U, 10)
        self.True_Obliquity = self.Mean_Obliquity/3600 + self.Obliquity_Nu

    def aberration_correction(self):
        self.Aberration_Correction = -20.4898 / (3600 * self.R_AU)

    def apparent_sun_longitude(self):
        self.aberration_correction()
        self.Apparent_Longitude = self.Geo_Longitude_deg + self.Longitude_Nu + self.Aberration_Correction

    def apparent_sideral_time_GreenWich(self):
        self.Mean_Sideral_Time = 280.46061837 + 360.98564736629 * (self.JulianDay - 2451545) + 0.000387933 * np.power(self.JulianCentury, 2) - (np.power(self.JulianCentury, 3)/38710000)
        #self.Mean_Sideral_Time = np.rad2deg(self.Mean_Sideral_Time
        F = self.Mean_Sideral_Time / 360
        F = np.modf(F)[0]
        piv = np.argwhere(self.Mean_Sideral_Time >= 0)
        niv = np.argwhere(self.Mean_Sideral_Time < 0)
        self.Mean_Sideral_Time[piv] = 360 * F[piv]
        self.Mean_Sideral_Time[niv] = (360 * F[niv]) + 360
        #self.Mean_Sideral_Time = np.rad2deg(self.Mean_Sideral_Time)
        self.Apparent_Sideral_Time = self.Mean_Sideral_Time + self.Longitude_Nu * np.cos(np.deg2rad(self.True_Obliquity))

    def geocentric_sun_right_ascension(self):
        self.Right_Ascension_rad = np.arctan2(np.sin(np.deg2rad(self.Apparent_Longitude)) * np.cos(np.deg2rad(self.True_Obliquity)) - np.tan(np.deg2rad(self.Geo_Latitude)) * np.sin(np.deg2rad(self.True_Obliquity)), np.cos(np.deg2rad(self.Apparent_Longitude)))
        self.Right_Ascension_deg = (self.Right_Ascension_rad * 180) / np.pi
        F = self.Right_Ascension_deg/360
        F = np.modf(F)[0]
        piv = np.argwhere(self.Right_Ascension_deg >= 0)
        niv = np.argwhere(self.Right_Ascension_deg < 0)
        self.Right_Ascension_deg[piv] = 360 * F[piv]
        self.Right_Ascension_deg[niv] = (360 * F[niv]) + 360

    def geocentric_declenation(self):
        self.Geo_Declenation_rad = np.arcsin(np.sin(np.deg2rad(self.Geo_Latitude)) * np.cos(np.deg2rad(np.deg2rad(self.True_Obliquity))) + np.cos(np.deg2rad(self.Geo_Latitude)) * np.sin(np.deg2rad(self.True_Obliquity)) * np.sin(np.deg2rad(self.Apparent_Longitude)))
        self.Geo_Declenation_deg = np.rad2deg(self.Geo_Declenation_rad)

    def observer_hour_angle(self):
        self.Observer_Hour_Angle_deg = self.Apparent_Sideral_Time + self.farm.Longitude_deg - self.Right_Ascension_deg
        F = self.Observer_Hour_Angle_deg/360
        F = np.modf(F)[0]
        piv = np.argwhere(self.Observer_Hour_Angle_deg >= 0)
        niv = np.argwhere(self.Observer_Hour_Angle_deg < 0)
        self.Observer_Hour_Angle_deg[piv] = 360 * F[piv]
        self.Observer_Hour_Angle_deg[niv] = (360 * F[niv]) + 360
        self.Observer_Hour_Angle_rad = np.deg2rad(self.Observer_Hour_Angle_deg)

    def topocentric_sun_right_ascension(self):
        self.Equatorial_horizontal_parallax = 8.764 / (3600 * self.R_AU)
        u = np.arctan(0.99664719 * np.tan(self.farm.Latitude_rad))
        x = np.cos(u) + self.farm.Elevation/6378140 * np.cos(self.farm.Latitude_rad)
        y = 0.99664719 * np.sin(u) + self.farm.Elevation/6378140 * np.sin(self.farm.Latitude_rad)
        self.parallax_right_ascension = np.arctan2(-x * np.sin(np.deg2rad(self.Equatorial_horizontal_parallax)) * np.sin(np.deg2rad(self.Observer_Hour_Angle_deg)), np.cos(self.Geo_Declenation_rad) - x * np.sin(np.deg2rad(self.Equatorial_horizontal_parallax) * np.cos(np.deg2rad(self.Observer_Hour_Angle_deg))))
        self.Topocentric_Sun_Right_Ascension = self.Right_Ascension_deg - np.rad2deg(self.parallax_right_ascension)
        self.Topocentric_Sun_Declenation_rad = np.arctan2((np.sin(self.Geo_Declenation_rad)-y*np.sin(np.deg2rad(self.Equatorial_horizontal_parallax))) * np.cos(np.deg2rad(self.parallax_right_ascension)), np.cos(self.Geo_Declenation_rad) - x * np.sin(np.deg2rad(self.parallax_right_ascension)) * np.cos(np.deg2rad(self.Observer_Hour_Angle_deg)))
        self.Topocentric_Sun_Declenation_deg = np.rad2deg(self.Topocentric_Sun_Declenation_rad)
        self.Topocentric_Local_Hour_Angle_deg = self.Observer_Hour_Angle_deg - np.rad2deg(self.parallax_right_ascension)
        self.Topocentric_Local_Hour_Angle_rad = np.deg2rad(self.Topocentric_Local_Hour_Angle_deg)

    def topocentirc_sun_zenith(self):
        self.Elevation_rad = np.arcsin(np.sin(self.farm.Latitude_rad) * np.sin(self.Topocentric_Sun_Declenation_rad) + np.cos(self.farm.Latitude_rad) * np.cos(self.Topocentric_Sun_Declenation_rad) * np.cos(self.Topocentric_Local_Hour_Angle_rad))
        self.Elevation_deg = np.rad2deg(self.Elevation_rad)
        self.Topocentric_Zenith_deg = 90 - self.Elevation_deg
        #self.Topocentric_Zenith_deg = np.where(self.Topocentric_Zenith_deg < 90, self.Topocentric_Zenith_deg, 0)
        self.Topocentric_Zenith_rad = np.deg2rad(self.Topocentric_Zenith_deg)

    def topocentric_sun_azimuth(self):
        self.Topocentric_Astronomers_Azimuth_rad = np.arctan2(np.sin(self.Topocentric_Local_Hour_Angle_rad),np.cos(self.Topocentric_Local_Hour_Angle_rad)*np.sin(self.farm.Latitude_rad) - np.tan(self.Topocentric_Sun_Declenation_rad) * np.cos(self.farm.Latitude_rad))
        self.Topocentric_Astronomers_Azimuth_deg = np.rad2deg(self.Topocentric_Astronomers_Azimuth_rad)
        self.Topocentric_Azimuth_deg = self.Topocentric_Astronomers_Azimuth_deg + 180
        self.Topocentric_Azimuth_rad = self.Topocentric_Azimuth_deg

    def equation_of_time(self):
        B = np.deg2rad(360 / 365 * (self.Day - 81))
        self.EOT = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
        return

    def declenation(self):
        self.Declenation_rad = -np.arcsin(0.39779 * np.cos(np.deg2rad(0.98565) * (self.Day + 10) + np.deg2rad(1.914) * np.sin(np.deg2rad(0.98565 * (self.Day - 2)))))
        #self.Declenation_rad = np.arcsin(np.sin(np.deg2rad(-23.45)) * np.cos(np.deg2rad(360/365 * (self.Day + 10))))
        self.Declenation_deg = np.rad2deg(self.Declenation_rad)
        return

    def zenith(self):
        self.Cos_SZA = np.sin(self.farm.Latitude_rad) * np.sin(self.Declenation_rad) + np.cos(self.farm.Latitude_rad) * np.cos(self.Declenation_rad) * np.cos(self.HourAngle_rad)
        self.Zenith_rad = np.real(np.arccos(self.Cos_SZA,dtype=np.complex))
        self.Zenith_deg = np.rad2deg(self.Zenith_rad)
        self.Altitude_deg = 90 - self.Zenith_deg
        self.Altitude_rad = np.deg2rad(self.Altitude_deg)
        return

    def azimuth(self):
        self.Cos_SSA = (np.sin(self.Declenation_rad) - np.cos(self.Zenith_rad) * np.sin(self.farm.Latitude_rad)) / (np.sin(self.Zenith_rad) * np.cos(self.farm.Latitude_rad))
        self.Azimuth_rad = np.real(np.arccos(self.Cos_SSA,dtype=complex))
        self.Azimuth_deg = np.rad2deg(self.Azimuth_rad)
        self.Azimuth_deg = 360 - self.Azimuth_deg
        return

    def air_mass(self):
        #self.AirMass = np.sqrt(np.power(708 * np.cos(self.Topocentric_Zenith_rad), 2) + 2 * 708 + 1) - 708 * np.cos(self.Topocentric_Zenith_rad)
        #self.AirMass = 1 / np.cos(self.Topocentric_Zenith_rad)
        self.AirMass = 1 / np.cos(self.Topocentric_Zenith_rad) + 0.50572 * np.power(96.07995 - self.Topocentric_Zenith_deg,-1.6364)
        #self.Airmass = np.abs(self.AirMass)
        self.AirMass = np.clip(self.AirMass, 0, 11)
        #self.AirMass = np.where(self.AirMass > 2, 2, self.AirMass)#+ 0.505272 * np.power((96.07995 - self.Topocentric_Zenith_deg),-1.6364))
        return

    def clearness(self):
        self.ClearnessIndex = (self.GHI / self.TOA) / (0.1 + 1.031 * np.exp(-1.4 / 0.9 + 9.4 / self.AirMass))
        #self.ClearnessIndex = np.clip(self.ClearnessIndex,0,1)
        return

    def vector_azimuth_and_zenith(self):
        lon_subsolar = np.deg2rad(- 15 * (self.Hour - 12 + self.EOT/60))
        self.lon_subsolar = lon_subsolar
        lat_subsolar = self.Topocentric_Sun_Declenation_rad
        self.lat_subsolar = lat_subsolar

        self.Sx = np.cos(lat_subsolar) * np.sin(lon_subsolar - self.farm.Longitude_rad)
        self.Sy = np.cos(self.farm.Latitude_rad) * np.sin(lat_subsolar) - np.sin(self.farm.Latitude_rad) * np.cos(lat_subsolar) * np.cos(lon_subsolar - self.farm.Longitude_rad)
        self.Sz = np.sin(self.farm.Latitude_rad) * np.sin(lat_subsolar) + np.cos(self.farm.Latitude_rad) * np.cos(lat_subsolar) * np.cos(lon_subsolar - self.farm.Longitude_rad)

        self.Zenith_Vec_rad = np.arccos(self.Sz)
        self.Zenith_Vec_deg = np.rad2deg(self.Zenith_Vec_rad)

        self.Azimuth_Vec_rad = np.arctan2(-self.Sx, -self.Sy)
        self.Azimuth_Vec_deg = np.rad2deg(self.Azimuth_Vec_rad)

        self.Zenith_deg = self.Zenith_Vec_deg.clip(0,90)
        self.Zenith_rad = np.deg2rad(self.Zenith_deg)
        self.Azimuth_deg = self.Azimuth_Vec_deg.clip(0,180)
        self.Azimuth_rad = self.Azimuth_Vec_rad
        self.Altitude_deg = 90 - self.Zenith_deg
        self.Altitude_rad = np.deg2rad(self.Altitude_deg)
        return


class Module:
    def __init__(self, Azimuth, Altitude, Farm, Sun):
        self.Azimuth_deg = Azimuth
        self.Azimuth_rad = np.deg2rad(self.Azimuth_deg)
        self.Altitude_deg = Altitude
        self.Altitude_rad = np.deg2rad(self.Altitude_deg)
        self.Farm = Farm
        self.Sun = Sun

    def Angle_Of_Incident(self):
        #self.d = self.Sun.Topocentric_Azimuth_deg - self.Azimuth_deg
        #d = np.deg2rad(self.d)
        self.AOI_rad = np.arccos(np.cos(self.Sun.Topocentric_Zenith_rad) * np.cos(self.Altitude_rad) + np.sin(self.Altitude_rad) * np.sin(self.Sun.Topocentric_Zenith_rad) * np.cos(self.Sun.Topocentric_Astronomers_Azimuth_rad - self.Azimuth_rad))
        self.AOI_deg = np.rad2deg(self.AOI_rad)
        self.AOI_deg = np.where(self.AOI_deg > 90,90,self.AOI_deg)
        self.AOI_rad = np.deg2rad(self.AOI_deg)
        return

    def Rays_Incident_Angle(self):

        self.IncidentAngle_rad = np.arccos(np.cos(self.Sun.Topocentric_Zenith_rad) * np.cos(self.Altitude_rad) + np.sin(self.Sun.Topocentric_Zenith_rad) * np.sin(self.Altitude_rad) * np.cos(self.Sun.Topocentric_Azimuth_rad - self.Azimuth_rad))
        #self.IncidentAngle_rad = np.cos(self.Sun.Azimuth_rad) * np.sin(self.Altitude_rad) * np.cos(self.Azimuth_rad - self.Sun.Azimuth_rad) + np.sin(self.Sun.Altitude_rad) * np.cos(self.Azimuth_rad)
        #self.IncidentAngle_rad = np.sin(self.Sun.Altitude_rad + self.Altitude_rad)
        self.IncidentAngle_deg = np.rad2deg(self.IncidentAngle_rad)
        return

    def Beam_Irradiance(self):
        self.BMult = np.cos(self.AOI_rad) #/ np.cos(self.Sun.Topocentric_Zenith_rad))
        self.BeamIrradiance = self.Sun.BNI * self.BMult#
        self.BeamIrradiance = np.nan_to_num(self.BeamIrradiance)
        #self.BeamIrradiance = np.where(self.BeamIrradiance < 0, 0, self.BeamIrradiance)

    def Diffuse_Irradiance(self):
        self.DiffuseIrradiance = np.zeros(len(self.Sun.JulianEphemerisCentury))

        Overcast = np.argwhere(self.Sun.ClearnessIndex < 0.3).ravel()
        Sunny = np.argwhere(self.Sun.ClearnessIndex >= 0.3).ravel()
        Low = np.argwhere(self.Sun.Elevation_rad < 0.1).ravel()

        self.DiffuseIrradiance[Overcast] = self.Sun.DHI[Overcast] * (((1 + np.cos(self.Altitude_rad))/2) + 0.25227 * (np.sin(self.Altitude_rad) - self.Altitude_rad*np.cos(self.Altitude_rad) - np.pi * np.power(np.sin(self.Altitude_rad/2),2)))

        self.DiffuseIrradiance[Sunny] = self.Sun.DHI[Sunny] * ((((1 + np.cos(self.Altitude_rad))/2) + (np.sin(self.Altitude_rad) - self.Altitude_rad*np.cos(self.Altitude_rad) - np.pi * np.power(np.sin(self.Altitude_rad/2),2))) * (0.00263 - 0.712 * (self.Sun.BHI[Sunny]/self.Sun.TOA[Sunny]) - 0.6883 * np.power(self.Sun.BHI[Sunny]/self.Sun.TOA[Sunny],2)) * (1 - (self.Sun.BHI[Sunny]/self.Sun.TOA[Sunny])) + ((self.Sun.BHI[Sunny]/self.Sun.TOA[Sunny]) * (np.cos(self.AOI_rad[Sunny])/np.cos(self.Sun.Topocentric_Zenith_rad[Sunny]))))

        self.DiffuseIrradiance[Low] = self.Sun.DHI[Low] * ((((1 + np.cos(self.Altitude_rad))/2) + (np.sin(self.Altitude_rad) - self.Altitude_rad*np.cos(self.Altitude_rad) - np.pi * np.power(np.sin(self.Altitude_rad/2),2))) * (0.00263 - 0.712 * (self.Sun.BHI[Low]/self.Sun.TOA[Low]) - 0.6883 * np.power(self.Sun.BHI[Low]/self.Sun.TOA[Low],2)) * (1 - (self.Sun.BHI[Low]/self.Sun.TOA[Low])) + ((self.Sun.BHI[Low]/self.Sun.TOA[Low]) * ((np.sin(self.Altitude_rad) * np.cos(self.Sun.Topocentric_Azimuth_rad[Low] - self.Azimuth_rad))/(0.1 - 0.008*self.Sun.Elevation_rad[Low]))))

        return

    def Perez_Diffuse_Irradiance(self):
        clearness_ratio = (self.Sun.DHI + self.Sun.BNI)/self.Sun.DHI
        top = clearness_ratio + (1.041 * np.power(self.Sun.Topocentric_Zenith_rad, 3))
        bottom = 1 + (1.041 * np.power(self.Sun.Topocentric_Zenith_rad, 3))
        self.Clearness = top/bottom
        #self.Clearness = ((self.Sun.DHI + self.Sun.BNI)/self.Sun.DHI + (1.041 * np.power(self.Sun.Topocentric_Zenith_rad,3))) / (1 + (1.041 * np.power(self.Sun.Topocentric_Zenith_rad,3)))
        self.Clearness_bin = self.Clearness

        self.Clearness_bin = np.where((self.Clearness > 1) & (self.Clearness <= 1.065), 1, self.Clearness)
        self.Clearness_bin = np.where((self.Clearness > 1.065) & (self.Clearness <= 1.230), 2, self.Clearness)
        self.Clearness_bin = np.where((self.Clearness > 1.230) & (self.Clearness <= 1.500), 3, self.Clearness)
        self.Clearness_bin = np.where((self.Clearness > 1.500) & (self.Clearness <= 1.950), 4, self.Clearness)
        self.Clearness_bin = np.where((self.Clearness > 1.950) & (self.Clearness <= 2.800), 5, self.Clearness)
        self.Clearness_bin = np.where((self.Clearness > 2.800) & (self.Clearness <= 4.500), 6, self.Clearness)
        self.Clearness_bin = np.where((self.Clearness > 4.500) & (self.Clearness <= 6.200), 7, self.Clearness)
        self.Clearness_bin = np.where(self.Clearness > 6.200, 8, self.Clearness)
        self.Delta = (self.Sun.DHI * self.Sun.AirMass) / self.Sun.TOA

        Coefficients = np.loadtxt('Perez/Coefficients.csv',delimiter=',',skiprows=1)

        f11 = Coefficients[np.nan_to_num((self.Clearness_bin - 1)).astype(int), 0]
        f12 = Coefficients[np.nan_to_num((self.Clearness_bin - 1)).astype(int), 1]
        f13 = Coefficients[np.nan_to_num((self.Clearness_bin - 1)).astype(int), 2]
        f21 = Coefficients[np.nan_to_num((self.Clearness_bin - 1)).astype(int), 3]
        f22 = Coefficients[np.nan_to_num((self.Clearness_bin - 1)).astype(int), 4]
        f23 = Coefficients[np.nan_to_num((self.Clearness_bin - 1)).astype(int), 5]


        self.F1 = f11 + (f12 * self.Delta) + (self.Sun.Topocentric_Zenith_rad * f13)
        self.F1 = np.maximum(0, self.F1)
        self.F1 = np.nan_to_num(self.F1)

        self.F2 = f21 + (f22 * self.Delta) + (np.pi * self.Sun.Topocentric_Zenith_deg)/180 * f23
        self.F2 = np.nan_to_num(self.F2)

        a = np.maximum(0, np.cos(self.AOI_rad))
        b = np.maximum(np.cos(np.deg2rad(85)), np.cos(self.Sun.Topocentric_Zenith_rad))


        self.c = (1 - self.F1) * (1 + np.cos(self.Altitude_rad))/2
        self.d = self.F1 * (a/b)
        self.e = self.F2 * np.sin(self.Altitude_rad)
        self.DiffuseIrradiance = self.Sun.DHI * (self.c + self.d + self.e)
        #self.DiffuseIrradiance = self.Sun.DHI * (((1 - self.F1) * (1 + np.cos(self.Altitude_rad)/2)) + self.F1 * (a/b) + self.F2 * np.sin(self.Altitude_rad))
        return

    def Reflected_Irradiance(self):
        self.ReflectedIrradiance = self.Sun.GHI * self.Farm.albedo * ((1 - np.cos(self.Altitude_rad)) / 2)
        return

    def Total_Irradiance(self):
        self.Irradiance = self.BeamIrradiance + self.DiffuseIrradiance + self.ReflectedIrradiance
        return

    def Fill_nan(self):
        self.Irradiance = np.nan_to_num(self.Irradiance)
        return

    def Temperature(self, NOCT):
        self.temperature = self.Farm.Temperature + (NOCT - 20)/80 * self.Irradiance
        return


class Huld:
    def __init__(self, module, coefficients_dir, STC_Irradiance, STC_Temperature, STC_Power):
        self.Module = module
        self.coefficients_dir = coefficients_dir
        self.STC_Irradiance = STC_Irradiance
        self.STC_Temperature = STC_Temperature
        self.STC_Power = STC_Power

    def load_coefficients(self):
        self.coefficients = np.loadtxt(self.coefficients_dir, delimiter=',', skiprows=1)

    def normalised_irradiance_and_Temperature(self):
        self.Normalised_Irradiance = self.Module.Irradiance / self.STC_Irradiance
        self.Normalised_Temperature = self.Module.temperature / self.STC_Temperature

    def module_power(self):
        self.power = self.Normalised_Irradiance * (self.STC_Power + (self.coefficients[0] * np.log(self.Normalised_Irradiance)) + (self.coefficients[1] * np.power(np.log(self.Normalised_Irradiance),2)) + (self.coefficients[2] * self.Normalised_Temperature) + (self.coefficients[3] * self.Normalised_Temperature * np.log(self.Normalised_Irradiance)) + (self.coefficients[4] * self.Normalised_Temperature * np.power(np.log(self.Normalised_Irradiance),2)) + (self.coefficients[5] * np.power(self.Normalised_Temperature,2)))
        self.power = np.nan_to_num(self.power)

if __name__ == "__main__":
    Raw = 'Greenwich_SolarTime_HourRes.csv'
    Data = Copernicus(Raw)
    Data.convert_time()
    Data.irradiation_components()

    #RandomLoc = Farm(51.47482, -0.000344, 0, 0.3, 'Weather/adaptor.mars.internal-1643632109.3332415-31597-16-51548f96-f1c6-46be-85b7-1b1f13942d46.grib')
    #RandomLoc.load_weather()
    #RandomLoc.save_weather()
    RandomLoc = Farm(51.47482, -0.000344, 0, 0.3)
    RandomLoc.load_weather_csv('Weather/Weatherdata.csv')

    RandomLoc_Sun = Sun(RandomLoc, Data)
    RandomLoc_Sun.timing()
    RandomLoc_Sun.heliocentric_properties()
    RandomLoc_Sun.geocentric_lat_long()
    RandomLoc_Sun.nutatiion_in_longitude_and_obliquity()
    RandomLoc_Sun.true_obliquity()
    RandomLoc_Sun.apparent_sun_longitude()
    RandomLoc_Sun.apparent_sideral_time_GreenWich()
    RandomLoc_Sun.geocentric_sun_right_ascension()
    RandomLoc_Sun.geocentric_declenation()
    RandomLoc_Sun.observer_hour_angle()
    RandomLoc_Sun.equation_of_time()
    RandomLoc_Sun.topocentric_sun_right_ascension()
    RandomLoc_Sun.topocentirc_sun_zenith()
    RandomLoc_Sun.topocentric_sun_azimuth()
    RandomLoc_Sun.air_mass()
    RandomLoc_Sun.clearness()

    ModuleLoc = Module(0, 36, RandomLoc, RandomLoc_Sun)
    ModuleLoc.Angle_Of_Incident()
    ModuleLoc.Beam_Irradiance()
    ModuleLoc.Diffuse_Irradiance()
    #ModuleLoc.Perez_Diffuse_Irradiance()
    ModuleLoc.Reflected_Irradiance()
    ModuleLoc.Total_Irradiance()
    ModuleLoc.Temperature(48)

    Power = Huld(ModuleLoc,'Devices/c-Si.csv',1000,25,300)
    Power.load_coefficients()
    Power.normalised_irradiance_and_Temperature()
    Power.module_power()

    z = ModuleLoc.Sun.ClearnessIndex.reshape(365,24)
    A = plt.imshow(z, interpolation=None, aspect='auto')
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Year")
    plt.colorbar(A, label='Beam Irradiance Wm^-2')
    plt.show()

