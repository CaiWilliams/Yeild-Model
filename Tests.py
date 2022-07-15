#from main import *

def Declenation():
    A = Module(0, 30, -70, -1)
    Days = np.linspace(0, 365, 366)
    Results = np.zeros(len(Days))
    EOT = np.zeros(len(Days))

    for idx, i in enumerate(Days):
        B = np.deg2rad(360 / 365 * (i - 81))
        EOT[idx] = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
        A.Solar_Zenith_angle(i, 22)
        Results[idx] = A.SolarzenithAngle_deg

    plt.plot(EOT, Results)
    plt.show()

def IncidentAngle():
    A = Module(0, 23, 60, -1)
    Days = np.linspace(0, 365, 366)
    Results = np.zeros(len(Days))
    EOT = np.zeros(len(Days))

    for idx, i in enumerate(Days):
        B = np.deg2rad(360 / 365 * (i - 81))
        EOT[idx] = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
        A.Rays_Incident_Angle(i, 10)
        A.Air_Mass()
        Results[idx] = A.AirMass

    plt.scatter(Days, Results)
    plt.show()

def JD():
    Year = 1582
    Month = 10
    Day = 15
    A = int(Year / 100)
    B = int(A / 4)
    C = int(2 - A + B)
    E = int(365.25 * (Year + 4716))
    F = int(30.6001 * (Month + 1))
    JulianDay = C + Day + E + F - 1524.5
    return JulianDay

print(JD())