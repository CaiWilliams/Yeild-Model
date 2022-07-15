from main import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def Load_PVGISData(dir):
    data = np.loadtxt(dir,delimiter=',',skiprows=1,usecols=1)
    P = data
    return P

def rmse(predictions, targets):
    return np.sqrt(np.power(np.sum(predictions - targets),2)/len(predictions))

PVGIS_Power = Load_PVGISData('2016/Timeseries_51.481_-0.001_SA_1kWp_crystSi_14_41deg_-3deg_2016_2016.csv')

Raw = 'Greenwich_SolarTime_HourRes.csv'#'2016/adaptor.cams_solar_rad2.retrieve-1643643707.393866-15636-13-77b46440-e438-41e5-912e-e6bcc86c82c2.csv'
Data = Copernicus(Raw)
Data.convert_time()
Data.irradiation_components()

#RandomLoc = Farm(51.47482, -0.000344, 0, 0.2, '2016/adaptor.mars.internal-1643648975.5187953-1888-3-33c1e127-82de-4c5b-86fc-49757f1f1def.grib')
#RandomLoc.load_weather()
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

ModuleLoc = Module(-3, 41, RandomLoc, RandomLoc_Sun)
ModuleLoc.Angle_Of_Incident()
ModuleLoc.Beam_Irradiance()
ModuleLoc.Diffuse_Irradiance()
#ModuleLoc.Perez_Diffuse_Irradiance()
ModuleLoc.Reflected_Irradiance()
ModuleLoc.Total_Irradiance()
ModuleLoc.Temperature(48)

Power = Huld(ModuleLoc,'Devices/c-Si.csv',1000,25,1000)
Power.load_coefficients()
Power.normalised_irradiance_and_Temperature()
Power.module_power()

# E = rmse(Power.power*(1-0.14),PVGIS_Power)
# print("RSME: " + str(E))
#
# R = r2_score(PVGIS_Power, Power.power*(1-0.14))
# print("R2: " + str(R))
#
# PE = (np.sum(Power.power*(1-0.14)) - np.sum(PVGIS_Power))/np.sum(PVGIS_Power) * 100
# print("Persentage Error: " + str(PE))
z1 = (Power.power*(1-0.14))

Power = Huld(ModuleLoc,'Devices/P3HTPCBM.csv',1000,25,1000)
Power.load_coefficients()
Power.normalised_irradiance_and_Temperature()
Power.module_power()
z2 = (Power.power*(1-0.14))

Power = Huld(ModuleLoc,'Devices/PM6D18L8BO.csv',1000,25,1000)
Power.load_coefficients()
Power.normalised_irradiance_and_Temperature()
Power.module_power()
z3 = (Power.power*(1-0.14))
# z = np.cumsum(z1) - np.cumsum(z2)
#
# print(z[-1]/1000)
# #plt.plot(RandomLoc_Sun.ClearnessIndex)
plt.plot(np.cumsum(z1)/1000)
plt.plot(np.cumsum(z2)/1000)
plt.plot(np.cumsum(z3)/1000)
print(np.sum(z1)/1000/1000)
print(np.sum(z2)/1000/1000)
print(np.sum(z3)/1000/1000)
plt.xlabel("Time step")
plt.ylabel("Energy Generated (kWh)")
#plt.scatter(np.sort(Power.power-(1-0.14)),np.sort(PVGIS_Power))
#plt.colorbar(A, label='Error Wm^-2')

plt.show()
