from main import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

Raw = 'Greenwich_SolarTime_HourRes.csv'
Data = Copernicus(Raw)
Data.convert_time()
Data.irradiation_components()

RandomLoc = Farm(51.47482, -0.000344, 0, 0.3)
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



fig = plt.figure(figsize=(7,7))
ax = plt.axes()
ModuleLoc = Module(0, 0, RandomLoc, RandomLoc_Sun)
ModuleLoc.Angle_Of_Incident()
ModuleLoc.Beam_Irradiance()
ModuleLoc.Perez_Diffuse_Irradiance()
ModuleLoc.Reflected_Irradiance()
ModuleLoc.Total_Irradiance()
z = ModuleLoc.Irradiance.reshape(365, 24)
im = plt.imshow(z,interpolation='none',aspect='auto')
plt.colorbar(im, label='Irradiance Wm^-2')
plt.xlabel("Minute of Day")
plt.ylabel("Day of Year")

def init():
    im.set_data(z)
    return [im]

def animate(i):
    ModuleLoc = Module(0, i, RandomLoc, RandomLoc_Sun)
    ModuleLoc.Angle_Of_Incident()
    ModuleLoc.Beam_Irradiance()
    ModuleLoc.Diffuse_Irradiance()
    ModuleLoc.Reflected_Irradiance()
    ModuleLoc.Total_Irradiance()
    z = ModuleLoc.Irradiance.reshape(365, 24)
    im.set_array(z)
    t = "Tilt Angle = " + str(i)
    plt.title(t)
    return [im]
    #A = plt.imshow(z, aspect='auto')
    #plt.xlabel("Hour of Day")
    #plt.ylabel("Day of Year")
    #plt.colorbar(A, label='Beam Irradiance Wm^-2')

anim = FuncAnimation(fig, animate,frames=361,interval=20,repeat=True,blit=True)
anim.save('PerezTilt.mp4',dpi=300)