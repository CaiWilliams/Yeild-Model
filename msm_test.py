import copy
import os

from main import *


def setup_loc(lat, lon, irradiance_dir, weather_dir):

    Data = Copernicus(irradiance_dir)
    Data.convert_time()
    Data.irradiation_components()

    RandomLoc = Farm(lat, lon, 0, 0.3)
    RandomLoc.load_weather_csv_2(weather_dir)

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
    
    tilt = lat * 0.87 + 3.1
    ModuleLoc = Module(0, tilt, RandomLoc, RandomLoc_Sun)
    ModuleLoc.Angle_Of_Incident()
    ModuleLoc.Beam_Irradiance()
    ModuleLoc.Diffuse_Irradiance()
    #ModuleLoc.Perez_Diffuse_Irradiance()
    ModuleLoc.Reflected_Irradiance()
    ModuleLoc.Total_Irradiance()
    ModuleLoc.Temperature(48)
    return ModuleLoc

def run(loc, device_dir):
    #loc = copy.deepcopy(loc)
    Power = Huld(loc, device_dir, 1000, 20, 1000)
    Power.load_coefficients()
    Power.normalised_irradiance_and_Temperature()
    Power.module_power()
    return Power.power*(1-0.14)

irradiance_dir = os.path.join(os.getcwd(),'Locations','Buxton','Buxton_Irradiance.csv')
weather_dir = os.path.join(os.getcwd(),'Locations','Buxton','Buxton_weather_clean.csv')
device_dir = os.path.join(os.getcwd(),'Devices','c-Si.csv')
device_dir2 = os.path.join(os.getcwd(),'Devices','P3HTPCBM.csv')
location = setup_loc(50.609502,-2.45751, irradiance_dir,weather_dir)
power = run(location,device_dir)
power2 = run(location,device_dir2)
plt.plot(power-power2)
plt.show()