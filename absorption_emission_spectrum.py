import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt 
from scipy.special import assoc_laguerre
from scipy.special import eval_genlaguerre
import scipy.constants as sc 
from math import *

# Constants
InverseMeter2Joule = sc.physical_constants['inverse meter-joule relationship'][0] #Unit:J
eV2J = sc.physical_constants['electron volt-joule relationship'][0] #Unit:J/eV
J2eV = sc.physical_constants['joule-electron volt relationship'][0] #Unit: eV/J
epsilon_0 = sc.physical_constants['electric constant'][0] #Unit:F m^-1
D2SI = 3.335640952 * (10 ** (-30)) #Unit:C*m
h_bar = sc.physical_constants['Planck constant over 2 pi'][0] #Unit:J*s
c = sc.physical_constants['speed of light in vacuum'][0] #Unit:m s^-1
k_B = sc.physical_constants['Boltzmann constant'][0] #Unit: J K^-1
wavenumber_to_frequency = c* 10**2 #cm/s
pi = np.pi

# fixed parameters
delta_E_eg = 4  # Electronic State Energy Difference(eV)
delta_Eeg = delta_E_eg* eV2J   # Electronic State Energy Difference (J)

freq = 500  # Optical Frequency (cm^-1)
omega0 = 2*pi*freq*wavenumber_to_frequency  # vibrational angular frequency (s^-1)
Energy_quantum = omega0*h_bar*J2eV  # Energy quantum = 0.0620 eV

N = 10000 # sample points
LadderNum = 50  # vibrational ladder number

energy_range = np.linspace(2, 5, N, endpoint = True)    # range (eV)
energy_list = energy_range*eV2J     # range (J)
omega = energy_range* eV2J/h_bar   # s^-1
Gamma = 10**(-22) # J

def FCoverlap(v1, v2, S):
    """ Franck-Condon overlap
    Args:
        v1: 2D np array
        v2: 2D np array
        S: int
    Return:
        2D np array, axis = (0, 1) = (v1, v2)
    """
    vlarge = np.maximum(v1, v2)
    vsmall = np.minimum(v1, v2)
    abs_diff = vlarge - vsmall
    return S**(abs_diff / 2) * np.exp(-S / 2) * np.sqrt(sp.factorial(vsmall) / sp.factorial(vlarge)) * eval_genlaguerre(vsmall, abs_diff, S)

def D(band_gap, omega):
    """ Lorentzian distribution
    band_gap: 2D np array
    omega: float
    """
    return 1/pi * Gamma / ((band_gap - omega)**2 + Gamma**2)

def Maxewll_Boltzmann_Distribution(ladders, T):
    """ Maxewll_Boltzmann_Distribution
    Args:
        ladders: 2D np array
        T: float
    Return:
        1D np array
    """
    if T == 0:
        return (ladders == 0).astype(int)
    return np.exp(-h_bar * ladders * omega0/k_B/T) * (1 - np.exp(-h_bar * omega0 / (k_B * T)))

def Absorption_coefficient(LadderNum, energy_list, S, T):
    '''generate energy level of each vibrational state'''
    ground_state_energy = (np.arange(float(LadderNum)) + 1/2) * (h_bar * omega0) # electronic ground state vibrational energy level    
    excited_state_energy = (np.arange(float(LadderNum)) + 1/2) * (h_bar * omega0) + delta_Eeg # electronic excited state vibrational energy level
    ground_state_energy, excited_state_energy = np.meshgrid(ground_state_energy, excited_state_energy)
    
    '''calculate the overall rate constant of each energy point'''
    nladder = np.arange(LadderNum)
    v1, v2 = np.meshgrid(nladder, nladder)
    I = [np.sum(FCoverlap(v1, v2, S)**2  * D(excited_state_energy - ground_state_energy, energy_list[i]) * Maxewll_Boltzmann_Distribution(v1, T), axis = (0, 1)) for i in range(len(energy_list))]
    I = np.array(I)
    return  I

def Emission_coefficient(LadderNum, energy_list, S, T):
    ground_state_energy = (np.arange(float(LadderNum)) + 1/2) * (h_bar * omega0) # electronic ground state vibrational energy level    
    excited_state_energy = (np.arange(float(LadderNum)) + 1/2) * (h_bar * omega0) + delta_Eeg # electronic excited state vibrational energy level
    ground_state_energy, excited_state_energy = np.meshgrid(ground_state_energy, excited_state_energy)
    
    nladder = np.arange(LadderNum)
    v1, v2 = np.meshgrid(nladder, nladder)
    I = [np.sum(FCoverlap(v1, v2, S)**2  * D(excited_state_energy - ground_state_energy, energy_list[i]) * Maxewll_Boltzmann_Distribution(v2, T), axis = (0, 1)) for i in range(len(energy_list))]
    I = np.array(I)
    return  I


# Plot series
print('Parameters:')
print('Electronic State Energy Difference = %.2f eV'%(delta_E_eg))
print('Optical Frequency = %.2f cm^(-1)'%(freq))
print('energy quantum = %.4f eV'%(Energy_quantum))
print('Line Width = %g J' %(Gamma))
print('_________________________________________________________________________________________________________')

# Huang Rhys Factor dependence
T = 0 # K
print('temperature = ', T, 'K')
S_list = list(range(10))
for S in S_list:
    print("S = ", S)
    absorption = Absorption_coefficient(LadderNum, energy_list, S, T)
    normalized_absorption = absorption/np.amax(absorption)
    plt.plot(energy_range, normalized_absorption)
    
    emission = Emission_coefficient(LadderNum, energy_list, S, T)
    normalized_emission = emission/np.amax(emission)
    plt.plot(energy_range, normalized_emission)
    
    plt.xlim([3,5])
    plt.xlabel('Energy (eV)')
    plt.ylabel('Normalized intensity')
    plt.legend(['Absorption', 'Emission'])
    plt.title('Simulated Absorption and Emission Spectroscopy S = %d, T = %d'%(S, T))
    plt.show()
    print('_________________________________________________________________________________________________________')


#Temperature dependence
S = 3   #Huang Rhys Factor
print("S = ", S)
T_list = [0, 10, 50, 100, 300, 1000]
for T in T_list:
    print("temperature = ", T, 'K')
    absorption = Absorption_coefficient(LadderNum, energy_list, S, T)
    normalized_absorption = absorption/np.amax(absorption)
    plt.plot(energy_range, normalized_absorption)
    
    emission = Emission_coefficient(LadderNum, energy_list, S, T)
    normalized_emission = emission/np.amax(emission)
    plt.plot(energy_range, normalized_emission)
    
    plt.xlim([3,5])
    plt.xlabel('Energy (eV)')
    plt.ylabel('Normalized intensity')
    plt.legend(['Absorption', 'Emission'])
    plt.title('Simulated Absorption and Emission Spectroscopy S = %d, T = %d'%(S, T))
    plt.show()
    print('_________________________________________________________________________________________________________')