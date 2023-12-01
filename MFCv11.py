# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:23:41 2020

@authors: miela gross and jackson bauer

#########################
Molecular Field Coefficients Model for rare-earth iron garnets
Updated: 12-1-2023

Based primarily on:
Dionne, G.F., Magnetic Moment Versus Temperature Curves of Rare-Earth Iron
Garnets, Technical Report (Lincoln Laboratory), 1979

See also:
Dionne, G., Molecular field coefficients of substituted yttrium iron garnets,
J. Appl. Phys. (41), 1970, 4874-4881

Dionne, G., Molecular-field coefficients of rare-earth iron garnets, J. Appl.
Phys. (47), 1976, 4220-4221

Wolf, W.P. and Van Vleck, J.H., Magnetism of europium garnet. Phys. Rev.
118(6), 1960, 1490.

Pauthenet, R., Les propriétés magnétiques des ferrites d’yttrium et de terres
rares de formule 5Fe2O3. 3M2O3. In Annales de Physique 13(3), 1958, 424-462

Geller, S., Williams, H.J., Sherwood, R.C., Remeika, J.P. and Espinosa, G.P.,
Magnetic study of the lighter rare-earth ions in the iron garnets. Phys. Rev.
131(3), 1963, 1080.

Geller, S., Remeika, J.P., Sherwood, R.C., Williams, H.J. and Espinosa, G.P.,
Magnetic study of the heavier rare-earth iron garnets. Phys. Rev. 137(3A),
1965, A1034.

Espinosa, G.P., Crystal chemical study of the rare‐earth iron garnets,
The Journal of Chemical Physics, 37(10), 1962, 2344-2347
#########################
"""

########## Import needed functions and packages

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import symbols, Eq, exp, nsolve
from scipy.optimize import fsolve
import os, sys

# Define hyperbolic cotangent
def coth(x):
    tanh_x = np.tanh(x)
    return np.where(tanh_x != 0, 1/tanh_x, np.inf)

# Define Brillouin function
def B(x,j):
    if x == 0:
        return 0
    else:
        return (2*j+1)/(2*j)*coth((2*j+1)/(2*j)*x)-(1/(2*j))*coth(x/(2*j))


def MFCmodel():
    ########## Collect user input
    
    print(' ')
    print('Enter the primary RE ion present')
    print(' ' )
    print('Note: model currently works for Y, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu '
          'and other nonmagnetic substitutions')
    
    RE = input('Primary rare earth ion:                           ')
    Csub = input('C/dodecahedral site substitution (y/n)?           ')
    Asub = input('A/octahedral site substitution (y/n)?             ')
    Dsub = input('D/tetrahedral site substitution (y/n)             ')
    
    if Asub == 'y' or Dsub == 'y':
        Fesub = input('Ion substituted for Fe (enter n if none):         ')
    else:
        Fesub = 'n'
    
    if Csub == 'y':
        Csubnum = input('Is there more than 1 C site substitution (y/n)?   ')
        if Csubnum == 'y':
            Csubion1 = input('1st C substituted ion:                            ')
            Csubamt1 = input('1st C substituted ion amount (x/3):               ')
            Csubion2 = input('2nd C substituted ion:                            ')
            Csubamt2 = input('2nd C substituted ion amount (x/3):               ')
            y = float(Csubamt1)/3
            z = float(Csubamt2)/3
            x = 1 - y - z
        elif Csubnum == 'n':
            Csubion = input('C substituted ion:                                ')
            Csubamt = input('C substituted ion amount (x/3):                   ')
            y = float(Csubamt)/3
            z = 0;
            x = 1 - y - z
    elif Csub == 'n':
        Csubamt = 0
        x = 1
        y = 0
        z = 0
    
    if Asub == 'y':
        Asubamt = input('A substitution amount (x/2):                      ')
        ka = float(Asubamt)/2
    elif Asub == 'n':
        Asubamt = 0
        ka = 0
    
    if Dsub == 'y':
        Dsubamt = input('D substitution amount (x/3):                      ')
        kd = float(Dsubamt)/3
    elif Dsub == 'n':
        Dsubamt = 0
        kd = 0
    
    print('')
    print('Model can incorporate off-stoichiometry (RE:(Fe+Sub) /= 0.6)')
    Compdata = input('Do you have composition data (XPS/WDS) (y/n):     ')
    if Compdata == 'y':
        refratio = float(input('Enter ratio of RE to Fe sites (RE/A+D):           '))
        if refratio > 0.6:
            num_C = 8 - 8 / (1 +refratio)
            excessRE = num_C - 3
            kaRE = excessRE / 2         # excess RE as frac of octahedral sites
            kc4 = 0
        elif refratio < 0.6:
            num_C = 8 - 8 / (1 + refratio)
            excessFe = 3 - num_C
            kc4 = excessFe / 3          # excess Fe as frac of dodecahedral sites
            kaRE = 0
        else:
            kaRE = 0
            kc4 = 0
    else:
        kaRE = 0
        kc4 = 0
        refratio = 0.6

    # Account for presence of Fe2+ for oxygen deficient garnets
    Fe2percent = input('Are there any Fe2+ in the a-sites? (y/n):         ')
    if Fe2percent == 'y':
        p = float(input('How much Fe2+ per formula unit? (x/2):            '))/2
    else:
        p = 0

    
    ########## Interpret user input
        
    rares = defaultdict(lambda : 0)
    rares['Y'] =  0
    rares['La'] = 1
    rares['Ce'] = 2
    rares['Pr'] = 3
    rares['Nd'] = 4
    rares['Pm'] = 5
    rares['Sm'] = 6
    rares['Eu'] = 7
    rares['Gd'] = 8
    rares['Tb'] = 9
    rares['Dy'] = 10
    rares['Ho'] = 11
    rares['Er'] = 12
    rares['Tm'] = 13
    rares['Yb'] = 14
    rares['Lu'] = 15
    
    Re1 = rares[RE]
    
    if (Csub == 'y') and (Csubnum == 'y'):
        Re2 = rares[Csubion1]
        Re3 = rares[Csubion2]
    elif (Csub == 'y') and (Csubnum == 'n'):
        Re2 = rares[Csubion]
        Re3 = 0
    elif (Csub == 'n'):
        Re2 = 0
        Re3 = 0
    
    
    ########## Parameters

    # Nij describes exchange coupling between i and j sublattices
    Nac = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : -3.44,
           9 : -4.2, 10 : -4.0, 11 : -2.1, 12 : -0.2, 13 : -1.0, 14 : -4.0, 15 : 0}
    Ncd = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 6.0,
           9 : 6.5, 10 : 6.0, 11 : 4.0, 12 : 2.2, 13 : 17.0, 14 : 8.0, 15 : 0}
    # c-sites have no nearest neighbors, so coupling within c sublattice is neglected
    Ncc = 0
    
    gc = {0 : 1, 1 : 1, 2 : 1, 3 : 1, 4 : 1, 5 : 1, 6 : 1, 7 : 1, 8 : 2, 9 : 1.73,
            10 : 1.54, 11 : 1.49, 12 : 1.38, 13 : 7/6, 14 : 8/7, 15 : 1}
    Jc = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 3.5, 9 : 6.0,
            10 : 7.5, 11 : 8.0, 12 : 7.5, 13 : 6.0, 14 : 3.5, 15 : 0}
    # Use effective angular momentum for system of equations
    Jcp = {0 : 1, 1 : 1, 2 : 1, 3 : 1, 4 : 1 , 5 : 1, 6 : 1, 7 : 3, 8 : 3.50,
            9 : 4.0, 10 : 4.6, 11 : 4.2, 12 : 4.0, 13 : 1.085, 14 : 1.49, 
            15 : 1}
    # Determines whether RE is magnetic/used in system of equations
    # Accounts for when g/J = 0, otherwise there's a div by 0 error
    coef = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 1, 9 : 1,
            10 : 1, 11 : 1, 12 : 1, 13 : 1, 14 : 1, 15 : 0}
    # Angstrom # Values from Espinosa J. Chem. Phys. 37 2344-2347 (1962) 
    # unknowns use YIG
    latparam = {0 : 12.376, 1 : 12.767, 2 : 12.376, 3 : 12.646, 4 : 12.600,
                5 : 12.376, 6 : 12.529, 7 : 12.498, 8 : 12.471, 9 : 12.436,
                10 : 12.405, 11 : 12.375, 12 : 12.347, 13 : 12.323, 14 : 12.302,
                15 : 12.283}
    
    a = (x*latparam[Re1] + y*latparam[Re2] + z*latparam[Re3])
    print('Lattice parameter currently is '+str(a)+' Angstroms')
    changea = input('Would you like to update? (y/n):                  ')
    if changea == 'y':
        a = float(input('What is the unit cell size in Angstrom?:            '))
    # Update latice parameter units to m
    a = a * 10**-10

    # n acts as scale factor which helps the solver
    na = 8*2/a**3   # a-site atoms per unit cell volume
    nd = 8*3/a**3   # d-site atoms per unit cell volume
    nc = 8*3/a**3   # c-site atoms per unit cell volume
    n = 8/a**3      # formula units per unit cell volume

    # Define constants
    NA = 6.02214*10**23                 # Avogadro's Number
    SIconv = NA*a**3/8/4/np.pi*10**6    # Solve for system of equations in SI units
    
    # Get MFC of each RE with the Fe a-site
    Nac1 = Nac[Re1]*SIconv
    Nac2 = Nac[Re2]*SIconv
    Nac3 = Nac[Re3]*SIconv

    # Get MFC of each RE with the Fe d-site
    Ncd1 = Ncd[Re1]*SIconv
    Ncd2 = Ncd[Re2]*SIconv
    Ncd3 = Ncd[Re3]*SIconv

    # Get landé g factor of each RE
    g1 = gc[Re1]
    g2 = gc[Re2]
    g3 = gc[Re3]

    # Get effective angular momentum of each RE
    J1 = Jcp[Re1]
    J2 = Jcp[Re2]
    J3 = Jcp[Re3]

    # Calculate MFCs of Fe sublattices
    # kd, ka, and kaRE here accounts for any defects in the a and d-sites
    Naa = -65*SIconv*(1 - 0.42*kd*3)
    Ndd = -30.4*SIconv*(1 - 0.43*ka*2 - 0.55*kaRE*2)
    Nad = 97*SIconv*(1 - 0.125*ka*2 - 0*kaRE*2 - 0.127*kd*3)


    # mca & mac N coefficients (inital guesses, change accordingly)
    Naca = -2.8*SIconv      # Start inital guess as Nac1
    Nacd = 8.3*SIconv       # Start inital guess as Ncd1
    Ncaa = -8.8*SIconv      # Start inital guess as Nac1
    Ncad = 17.6*SIconv      # Start inital guess as Ncd1
    
    ########## Calculate Recursively
    
    # When composition data is supplied, the excess R.E. sits on the octahedral 
    # sublattice.  Naa1 and Nad1 are the MFC coef. for the R.E., and are
    # assumed to be the same as Fe.  This is justified, because the sites
    # should be approximately the same size and the bond lengths are also
    # similar.  This proved to be easier for the model to converge, and is a
    # more realistic assumption.  Distortion in size and position of the
    # lattice site are likely, but will require further study.

    # Define g-factor and J for Fe3+ and Fe2+ (only consider spin for Fe)
    g_Fe = 2.0
    J_Fe = 5/2
    J_Fe2 = 2.0

    # Define constants
    muB = 9.274 * 10**-24   # erg/G     A/m^2   J/T
    kB = 1.3806 * 10**-23   # erg/K     J/K
    mu0 = 1.2566 * 10**-6   # kg*m/(s^2 * A^2)
    
    # Create coefficients to simplify calculations
    const_Fe = g_Fe*muB*J_Fe*mu0
    const_c1 = g1*muB*J1*mu0*coef[Re1]
    const_c2 = g2*muB*J2*mu0*coef[Re2]
    const_c3 = g3*muB*J3*mu0*coef[Re3]
    gc = g1*x*coef[Re1]+g2*y*coef[Re2]+g3*z*coef[Re3]
    Jc = J1*x+J2*y+J3*z
    const_c = gc*muB*Jc*mu0
    
    # Zero temperature magnetic moments for each ion and site (muB/formula unit)
    Md0 = nd*g_Fe*J_Fe*muB*(1-kd)*(1-0.1*(ka+kaRE))
    Ma0 = na*g_Fe*(J_Fe*(1-p)+J_Fe2*p)*muB*(1-ka-kaRE)*(1-kd**5.4)
    Mac0 = kaRE*na*gc*Jc*muB*(1-kd**5.4)
    Mc10 = (1-kc4)*nc*x*g1*J1*muB*coef[Re1]
    Mc20 = (1-kc4)*nc*y*g2*J2*muB*coef[Re2]
    Mc30 = (1-kc4)*nc*z*g3*J3*muB*coef[Re3]
    Mca0 = kc4*nc*g_Fe*J_Fe*muB
    Mnet0 = np.abs(Md0 - Ma0 - Mac0 - Mc10 - Mc20 - Mc30 - Mca0)
    
    # Define temperature vector
    tlow = 1
    thigh = 600
    tpoints = 600
    Temp = np.linspace(tlow, thigh, num = tpoints)
    
    # Pre-allocate the moment vectors
    Mag = np.zeros(tpoints)
    Magd = np.zeros(tpoints)
    Maga = np.zeros(tpoints)
    Magac = np.zeros(tpoints)
    Magc1 = np.zeros(tpoints)
    Magc2 = np.zeros(tpoints)
    Magc3 = np.zeros(tpoints)
    Magca = np.zeros(tpoints)

    # Initial Guess
    x0 = np.array([Ma0, Mac0, Md0, Mc10, Mc20, Mc30, Mca0])

    # Account for zero division error when calculating Jac
    # this does not impact M since Mac0 is already 0
    if gc == 0:
        gc = 1

    # Iterate through temperature vector to solve for moment vectors
    for T in Temp:
        k, = np.where(Temp == T)
        i = k[0]

        # Define symbols
        ma,mac,md,mc1,mc2,mc3,mca = symbols('ma mac md mc1 mc2 mc3 mca')

        # Define equations
        def equations(x):
            ma, mac, md, mc1, mc2, mc3, mca = x
            eq1 = ma - Ma0 * B(const_Fe*(Naa*ma+Naca*mac+Nad*md+Nac1*mc1+Nac2*mc2+Nac3*mc3)/(kB*T),J_Fe)
            eq2 = mac - Mac0 * B(const_c*(Naca*ma+Nacd*md)/(kB*T),Jc)
            eq3 = md - Md0 * B(const_Fe*(Ndd*md+Nad*ma+Nacd*mac+Ncd1*mc1+Ncd2*mc2+Ncd3*mc3)/(kB*T),J_Fe)
            eq4 = mc1 - Mc10 * B(const_c1*(Nac1*ma+Ncd1*md+Ncc*mc1)/(kB*T),J1)
            eq5 = mc2 - Mc20 * B(const_c2*(Nac2*ma+Ncd2*md+Ncc*mc2)/(kB*T),J2)
            eq6 = mc3 - Mc30 * B(const_c3*(Nac3*ma+Ncd3*md+Ncc*mc3)/(kB*T),J3)
            eq7 = mca - Mca0 * B(const_Fe*(Ncaa*ma+Ncad*md)/(kB*T),J_Fe)
            return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]

        # Initial Guess
        x0 = np.array([Ma0, Mac0, Md0, Mc10, Mc20, Mc30, Mca0])
    
        # Solve the system of equations using fsolve in Scipy library
        Sol = fsolve(equations, x0)

        Ma, Mac, Md, Mc1, Mc2, Mc3, Mca = Sol

        # Input moments in units of muB
        Magd[int(i)] = Md/muB/n
        Maga[int(i)] = Ma/muB/n
        Magac[int(i)] = Mac/muB/n
        Magc1[int(i)] = Mc1/muB/n
        Magc2[int(i)] = Mc2/muB/n
        Magc3[int(i)] = Mc3/muB/n
        Magca[int(i)] = Mca/muB/n
        Mag[int(i)] = abs(Md-Ma-Mac-Mc1-Mc2-Mc3-Mca)/muB/n

        # Set Next Initial Guess
        x0 = np.array([Ma, Mac, Md, Mc1, Mc2, Mc3, Mca])

    # Insert zero temperature values at beginning of vectors
    Magd = np.insert(Magd, 0, Md0/muB/n, axis = 0)
    Maga = np.insert(Maga, 0, Ma0/muB/n, axis = 0)
    Magac = np.insert(Magac, 0, Mac0/muB/n, axis = 0)
    Magc1 = np.insert(Magc1, 0, Mc10/muB/n, axis = 0)
    Magc2 = np.insert(Magc2, 0, Mc20/muB/n, axis = 0)
    Magc3 = np.insert(Magc3, 0, Mc30/muB/n, axis = 0)
    Magca = np.insert(Magca, 0, Mca0/muB/n, axis = 0)
    Mag = np.insert(Mag, 0, Mnet0/muB/n, axis = 0)
    Temp = np.insert(Temp, 0, 0, axis = 0)

    # Account for Eu induced magnetization from Fe d-site coupling
    # Wolf W. P. Phys. Rev. 118.6 (1960)
    if Re1 == 7:
        Magc1 = [Magd[i]*0.162*x/(1+3*np.exp(-480/Temp[i])) for i in range(tpoints+1)]
        Mag = [Mag[i]-Magc1[i] for i in range(tpoints+1)]
    if Re2 == 7:
        Magc2 = [Magd[i]*0.162*y/(1+3*np.exp(-480/Temp[i])) for i in range(tpoints+1)]
        Mag = [Mag[i]-Magc2[i] for i in range(tpoints+1)]
    if Re3 == 7:
        Magc3 = [Magd[i]*0.162*z/(1+3*np.exp(-480/Temp[i])) for i in range(tpoints+1)]
        Mag = [Mag[i]-Magc3[i] for i in range(tpoints+1)]

    # Account for Pr magnetization, linear to 1.04 of the net Fe moment and
    # coupled ferromagnetically to d-site Geller S. Phys Rev 131 3 (1963)
    if Re1 == 3:
        Magc1 = [(Magd[i]-Maga[i])*1.04*x for i in range(tpoints+1)]
        Mag = [Mag[i]+Magc1[i] for i in range(tpoints+1)]
    if Re2 == 3:
        Magc2 = [(Magd[i]-Maga[i])*1.04*y for i in range(tpoints+1)]
        Mag = [Mag[i]+Magc2[i] for i in range(tpoints+1)]
    if Re3 == 3:
        Magc3 = [(Magd[i]-Maga[i])*1.04*z for i in range(tpoints+1)]
        Mag = [Mag[i]+Magc3[i] for i in range(tpoints+1)]

    # Create and fill angular momentum vectors
    Jd = [elem/g_Fe for elem in Magd]
    Ja = [elem/g_Fe for elem in Maga]
    Jac = [elem/gc for elem in Magac]
    Jc1 = [elem/g1 for elem in Magc1]
    Jc2 = [elem/g2 for elem in Magc2]
    Jc3 = [elem/g3 for elem in Magc3]
    Jca = [elem/g_Fe for elem in Magca]
    J = [abs(Magd[i]/g_Fe-Maga[i]/g_Fe-Magac[i]/gc-Magc1[i]/g1-Magc2[i]/g2-Magc3[i]/g3-Magca[i]/g_Fe) for i in range(tpoints+1)]

 
    #### Calculate room temperature magnetization, compensation temperature,
    # and Curie temperature
    rtmag = (Mag[300])       # muB
    print(' ')
    print('300 K magnetic moment:    ',round(rtmag, 2),' muB')

    index = 0 # Index of curie temperature
    # Define Curie temperature as when Mag < 0.001 muB/formula unit
    for i in np.arange(-600,0,1):
        if Mag[-i] > 0.001:
            tcurie = Temp[-i]
            index = -i
            break
    # Only Gd, Tb, Dy, Ho, and Er have bulk compensation temperatures
    if Re1 == 8 or Re1 == 9 or Re1 == 10 or Re1 == 11 or Re1 == 12:
        tcomp = np.argmin(Mag[0:index])
        tang = np.argmin(J[0:index])
        print('Magnetization compensation temperature:     ', round(tcomp, 2),' K')
        print('Angular momentum compensation temperature:  ', round(tang, 2),' K')
        
    print('Zero temperature moment:  ',round(Mag[0], 2),' muB')
    print('Curie temperature:        ', round(tcurie, 2), ' K')
    print('kc4 = ', kc4)
    print('kaRE = ', kaRE)
    
    #### Plot output data

    fig, (ax1, ax2) = plt.subplots(1,2)
    
    # Create chemical formula for Legend
    if Compdata == 'y' and refratio < 0.6:
        if Csub == 'n':
            fe_amt = round(3*kc4, 2)
            c_re = round(3 - fe_amt, 2)
            dodecsite = ('{$' + RE + '_{' + str(c_re) + '}$$' + 'Fe_{' + 
                          str(fe_amt) + '}$}') 
        elif Csub == 'y' and Csubnum == 'n':
            re1_amt = round(3*x*(1-kc4), 2)
            re2_amt = round(3*y*(1-kc4), 2)
            fe_amt = round(3*kc4, 2)
            dodecsite = ('{$' + RE + '_{' + str(re1_amt) + '}$$' + Csubion 
                         + '_{' + str(re2_amt) + '}$$' + 'Fe_{' + str(fe_amt) 
                         + '}$}')
            ax1.plot(Temp, Magc2, label = '$M_{c,'+Csubion+'}$', linewidth = 4)
            ax2.plot(Temp, Jc2, label = '$J_{c,'+Csubion+'}$', linewidth = 4)
        elif Csub == 'y' and Csubnum == 'y':
            re1_amt = round(3*x*(1-kc4), 2)
            re2_amt = round(3*y*(1-kc4), 2)
            re3_amt = round(3*z*(1-kc4), 2)
            fe_amt = round(3*kc4, 2)
            dodecsite = ('{$' + RE + '_{' + str(re1_amt) + '}$$' + Csubion1
                         + '_{' + str(re2_amt) + '}$$' + Csubion2 + '_{' 
                         + str(re3_amt) + '}$$' + 'Fe_{' + str(fe_amt) 
                         + '}$}')
            ax1.plot(Temp, Magc2, label = '$M_{c,'+Csubion1+'}$', linewidth = 4)
            ax1.plot(Temp, Magc3, label = '$M_{c,'+Csubion2+'}$', linewidth = 4)
            ax2.plot(Temp, Jc2, label = '$J_{c,'+Csubion1+'}$', linewidth = 4)
            ax2.plot(Temp, Jc3, label = '$J_{c,'+Csubion2+'}$', linewidth = 4)
    else:
        if Csub == 'n':
            dodecsite = ('{$' + RE + '_{' + str(3) + '}$}') 
        elif Csub == 'y' and Csubnum == 'n':
            re1_amt = round(3*x, 2)
            re2_amt = round(3*y, 2)
            dodecsite = ('{$' + RE + '_{' + str(re1_amt) + '}$$' + Csubion 
                         + '_{' + str(re2_amt) + '}$}')
            ax1.plot(Temp, Magc2, label = '$M_{c,'+Csubion+'}$', linewidth = 4)
            ax2.plot(Temp, Jc2, label = '$J_{c,'+Csubion+'}$', linewidth = 4)
        elif Csub == 'y' and Csubnum == 'y':
            re1_amt = round(3*x, 2)
            re2_amt = round(3*y, 2)
            re3_amt = round(3*z, 2)
            dodecsite = ('{$' + RE + '_{' + str(re1_amt) + '}$$' + Csubion1
                         + '_{' + str(re2_amt) + '}$$' + Csubion2 + '_{' 
                         + str(re3_amt) +'}$}')
            ax1.plot(Temp, Magc2, label = '$M_{c,'+Csubion1+'}$', linewidth = 4)
            ax1.plot(Temp, Magc3, label = '$M_{c,'+Csubion2+'}$', linewidth = 4)
            ax2.plot(Temp, Jc2, label = '$J_{c,'+Csubion1+'}$', linewidth = 4)
            ax2.plot(Temp, Jc3, label = '$J_{c,'+Csubion2+'}$', linewidth = 4)
        
    if Compdata == 'y'and refratio > 0.6:
        if Asub == 'y':
            a_amt = round(2*ka, 2)
            re_amt = round(2*kaRE, 2)
            a_fe = round(2 - a_amt - re_amt,2)
            asite = ('[$' + RE + '_{' + str(re_amt) +'}$$' + 'Fe_{' + 
                     str(a_fe) + '}$$' + Fesub + '_{' + str(a_amt) + '}$]')
        elif Asub == 'n':
            re_amt = round(2*kaRE, 2)
            a_fe = round(2 - re_amt,2)
            asite = ('[$' + RE + '_{' + str(re_amt) + '}$$' + 'Fe_{' + str(a_fe) +
                     '}$]')
    else:
        if Asub == 'y':
            a_amt = round(2*ka, 2)
            a_fe = round(2 - a_amt, 2)
            asite = ('[$' + 'Fe_{' + str(a_fe) +'}$$' + Fesub + '_{' + str(a_amt) +
                     '}$]')
        elif Asub == 'n':
            asite = '[$Fe_{2}$]'
    
    if Dsub == 'y':
        d_amt = round(3*kd, 2)
        d_fe = round(3 - d_amt, 2)
        dsite = ('($' + 'Fe_{' + str(d_fe) +'}$$' + Fesub + '_{' + str(d_amt) +
                 '}$)')
    elif Dsub == 'n':
        dsite = '($Fe_{3}$)'

    if refratio < 0.6:
        ax1.plot(Temp, Magca, label = '$M_{c,Fe}$', linewidth = 4)
        ax2.plot(Temp, Jca, label = '$J_{c,Fe}$', linewidth = 4)
    elif refratio > 0.6:
        ax1.plot(Temp, Magac, label = '$M_{a,'+RE+'}$', linewidth = 4)
        ax2.plot(Temp, Jac, label = '$J_{a,'+RE+'}$', linewidth = 4)
        
    chemformula = dodecsite + asite + dsite + '$O_{12}$'

    # Get directory for saving figures and data
    directory = ''
    dire = input('Directory for saving figures and data (c for current directory):                           ')
    if dire == 'c' or 'C':
        directory = os.getcwd()
    elif dire[-1] != '/':
        directory = dire + '/'
    else:
        directory = dire

    # Plot Momement and Angular Momentum Sublatttices over Temperature
    ax1.plot(Temp, Maga, label = '$M_{a,Fe}$', linewidth = 4)
    ax1.plot(Temp, Magd, label = '$M_{d,Fe}$', linewidth = 4)
    ax1.plot(Temp, Magc1, label = '$M_{c,'+RE+'}$', linewidth = 4)
    ax1.plot(Temp, Mag, label = '$M_s$', linewidth = 4)
    ax1.set_xlabel('Temperature (K)', fontsize = 28)
    ax1.set_ylabel('Magnetization ($\mu_B$)', fontsize = 28)
    ax1.set_title('Molecular Field Coefficients Model', fontsize = 32)
    ax1.set_xlim(0, thigh)
    ax1.set_ylim(0, (np.array([Mag,Maga,Magd,Magc1,Magc2,Magc3])).max()*1.1)
    ax1.legend(fontsize = 24, title = chemformula, title_fontsize = 24)
    
    ax2.plot(Temp, Ja, label = '$J_{a,Fe}$', linewidth = 4)
    ax2.plot(Temp, Jd, label = '$J_{d,Fe}$', linewidth = 4)
    ax2.plot(Temp, Jc1, label = '$J_{c,'+RE+'}$', linewidth = 4)
    ax2.plot(Temp, J, label = '$J$', linewidth = 4)
    ax2.set_xlabel('Temperature (K)', fontsize = 28)
    ax2.set_ylabel('Angular Momentum ($\hbar$)', fontsize = 28)
    ax2.set_xlim(0, thigh)
    ax2.set_ylim(0, (np.array([J,Ja,Jd,Jac,Jc1,Jc2,Jc3,Jca])).max()*1.1)
    ax2.legend(fontsize = 24, title = chemformula, title_fontsize = 24)

    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    fig.set_size_inches(20,8)

    name = input('Name for saving figures:                           ')
    chemform = [chemformula]
    plt.savefig(directory+name+'.png', bbox_inches='tight')

    # Plot Moment and Angular Momentum versus Temp together
    fig2, ax = plt.subplots()
    ax.plot(Temp, Mag, label = '$M_s$', linewidth = 4, color='tab:red')
    ax.set_xlabel('Temperature (K)', fontsize = 28)
    ax.set_ylabel('Magnetization ($\mu_B$)', fontsize = 28, color='tab:red')
    ax.set_title(chemformula, fontsize = 32)
    ax.set_xlim(0, thigh)
    ax.set_ylim(0, (np.array(Mag)).max()*1.1)
    ax.tick_params(axis='y', labelcolor='tab:red')

    axx = ax.twinx()
    axx.plot(Temp, J, label = '$J$', linewidth = 4,  color='tab:blue')
    axx.set_ylabel('Angular Momentum ($\hbar$)', fontsize = 28,  color='tab:blue')
    axx.set_ylim(0, (np.array([J])).max()*1.1)
    axx.tick_params(axis='y', labelcolor='tab:blue')

    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    axx.yaxis.set_tick_params(labelsize=20)
    fig2.set_size_inches(12,8)
    plt.savefig(directory+name+'TaTc.png', bbox_inches='tight')


    # Plot Ms versus Temp (no sublattices)
    fig3, ax = plt.subplots()
    ax.plot(Temp, Mag, label = '$M_s$', linewidth = 4)
    ax.set_xlabel('Temperature (K)', fontsize = 28)
    ax.set_ylabel('Magnetization $(\mu_B)$', fontsize = 28)
    ax.set_title(chemformula, fontsize = 32)
    ax.set_xlim(0, thigh)
    ax.set_ylim(0, (np.array(Mag)).max()*1.1)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    axx.yaxis.set_tick_params(labelsize=20)
    fig3.set_size_inches(10,8)
    plt.savefig(directory+name+'Ms.png', bbox_inches='tight')


    #Save array to file
    datasave = [[Temp[elem], Mag[elem]] for elem in range(len(Temp))]
    datasave = np.array(datasave)
    np.savetxt(directory+name+'data.txt', datasave)

    return

MFCmodel()
