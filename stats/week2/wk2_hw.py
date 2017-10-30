#!/usr/bin/python3

import numpy as np
import math
import matplotlib.pyplot as plt

SEP='-'*37

def q2():
    print(SEP, 'Q2', SEP)

    # constants
    R = 8.3145
    A_vdw = 1.408e-1
    B_vdw = 3.913e-5

    def P_vdw(mv, t):
        return R * t / (mv - B_vdw) - A_vdw / mv**2

    # find best estimate, error in two parameters, and overall error
    def do_calcs(mv, t, amv, at):
        # best estimate of P
        p_be = P_vdw(mv, t)
        print('Best estimate for pressure is: {:.5g} MPa'.format(p_be/1e6))

        # error in molar volume
        p_alpha_mv = P_vdw(mv +amv, t)
        err_mv = abs(p_alpha_mv - p_be)
        print('Error in P due to molar volume is {:.5g} MPa'.format(err_mv/1e6))
    
        # error in temperature
        p_alpha_t = P_vdw(mv, t + at)    
        err_t = abs(p_alpha_t - p_be)
        print('Error in P due to temperature is {:.5g} MPa'.format(err_t/1e6))

        # add errors in quadrature
        err = math.sqrt(err_mv**2 + err_t**2)
        print('Overall uncertainty in P is: {:.5g} MPa'.format(err/1e6))

        print('Final result for P: {:.4g} +- {:.1g} MPa'.format(p_be/1e6, err/1e6))

    do_calcs(mv=2e-4,t=298.0,amv=0.003e-4,at=0.2)
    do_calcs(mv=2e-3,t=400.0,amv=0.003e-3,at=0.2)

def q3():
    print(SEP, 'Q3', SEP)

    A = 7.32
    B = 29
    C = 22

    # use "cosine rule" to calculate angle from three lengths
    def calc_angle(a, b, c):
        return math.acos((b**2+c**2-a**2)/(2*b*c))

    # best estimate for angle
    angle_be = calc_angle(A, B, C)

    # uncertainty in measurement of lengths
    alpha_l = 0.1

    angle_err = 1
    alpha_ang = 1
    # to quote angle to 11 significant digits
    # need relative error < 1e-11
    while alpha_ang / angle_err > 1e-11:
        angle_err = calc_angle(A + alpha_l, B + alpha_l, C + alpha_l)
        alpha_ang = abs(angle_err - angle_be)
        
        # very gradually reduce error in length
        # hacky, should do proper iterative trial and improvement
        # but does converge
        alpha_l *= 0.99999
        
    print('To quote angle to 11s.f., need to know lengths to relative precision {:.1g}'
          .format(alpha_l))

# Perform weighted least-squares fit
def calcfit(xs, ys, errs):
    weights = errs**(-2)

    sumweights = sum(weights)
    weighted_x = sum(xs * weights)
    weighted_y = sum(ys * weights)
    weighted_xy = sum(xs * ys * weights)
    weighted_xsq = sum(xs**2 * weights)
    
    delta_prime = sumweights * weighted_xsq - weighted_x**2
    c = (weighted_xsq * weighted_y - weighted_x * weighted_xy) / delta_prime
    m = (sumweights * weighted_xy - weighted_x * weighted_y) / delta_prime
    alpha_c = np.sqrt(weighted_xsq / delta_prime)
    alpha_m = np.sqrt(sumweights / delta_prime)

    return m, c, alpha_m, alpha_c

def q4and5():
    print(SEP, 'Q4', SEP)

    freqs = np.linspace(10, 110, 11)
    voltages = np.array((16, 45, 64, 75, 70, 115, 142, 167, 183, 160, 221),
                        dtype='f') * 1e-3
    errors = np.array((5, 5, 5, 5, 30, 5, 5, 5, 5, 30, 5), dtype='f') * 1e-3
    weights = errors**(-2)
    
    m, c, alpha_m, alpha_c = calcfit(freqs, voltages, errors)
    
    print("""Best-fit gradient: {:.3g} mV/Hz
Best-fit intercept: {:.0f} mV
Error in gradient: {:.1g} mV/Hz
Error in intercept: {:.1g} mV"""
          .format(m * 1e3, c * 1e3, alpha_m * 1e3, alpha_c * 1e3))

    print(SEP, 'Q5', SEP)

    def chi_sq(xs, ys, weights, my_m, my_c):
        return sum(weights * (ys - my_m * xs - my_c)**2)

    # Use Newton-Raphson iteration to minimise the partial derivatives
    from scipy.optimize import newton

    def dSdm(my_m, my_c):
        func = lambda x: my_m * x + my_c
        return -2. * sum(weights * freqs * (voltages - list(map(func, freqs))))

    def dSdc(my_c, my_m):
        func = lambda x: my_m * x + my_c
        return -2. * sum(weights * (voltages - list(map(func, freqs))))
    
    minimised_m = newton(dSdm, 1, args=(c,))
    minimised_c = newton(dSdc, 1, args=(m,))

    minimised_chi_sq = chi_sq(freqs, voltages, weights, minimised_m, minimised_c)
    
    print("""Gradient minimising chi^2: {:.3g} mV/Hz
Intercept minimising chi^2: {:.0f} mV"""
          .format(minimised_m * 1e3, minimised_c * 1e3))
    print('These values agree with those obtained from the least-squares method above')

    # procedure for finding errors in m and c from chi^2 + 1
    # 'which' is index of param to find error for: 0 for m, 1 for c
    # this is also hacky but also works
    def finderr(which):
        params = [minimised_m, minimised_c]
        funcs = [dSdm, dSdc]
        tol = 1e-6
        new_chi_sq = 10
        old_chi_sq = 1
        assert(which < 2)
        results = [params.copy()]

        while abs((new_chi_sq / old_chi_sq) - 1) > tol:
            old_chi_sq = chi_sq(freqs, voltages, weights, *params)
            # modify current parameter to reach chi^2_min + 1
            while (chi_sq(freqs, voltages, weights, *params) - minimised_chi_sq) < 1:
                params[which] *= 1.0001
            results.append(params.copy())
            # optimise other parameter
            params[1 - which] = newton(funcs[1 - which], params[1 - which], args=(params[which],))
            results.append(params.copy())
            new_chi_sq = chi_sq(freqs, voltages, weights, *params)
        return np.array(results), abs(params[which] - results[0][which])

    # m error
    m_res, m_err = finderr(0)
    print('Error in gradient: {:.1g} mV/Hz'.format(m_err * 1e3))
    print('First five steps result in m and c as follows:')
    print(m_res[0:5])

    # c error
    (c_res, c_err) = finderr(1)
    print('Error in intercept: {:.1g} mV'.format(c_err * 1e3))
    print('These values also agree with those obtained from the least-squares method')
    
    N = 100
    ms = np.linspace(2.0e-3, 2.1e-3, N)
    cs = np.linspace(-4.5e-3, 0.5e-3, N)
    chisqs = np.zeros((N, N))
    for x in range(N):
        for y in range(N):
            chisqs[y, x] = chi_sq(freqs, voltages, weights, ms[x], cs[y]) - minimised_chi_sq
            
    plt.figure()
    ax = plt.gca()
    ax.set_xlabel('Gradient $m$ (V/Hz)')
    ax.set_ylabel('Intercept $c$ (V)')
    ax.set_title('$\Delta\chi^2$ minimisation for fitting $y=mx+c$ to data')
    plt.contourf(ms, cs, chisqs, range(0,11,1))
    plt.plot(m_res[:,0], m_res[:,1], label='Gradient error calculation')
    plt.plot(c_res[:,0], c_res[:,1], color='r', label='Intercept error calculation')
    cb = plt.colorbar()
    leg = plt.legend()
    cb.set_label('$\Delta\chi^2$')
    plt.savefig('chi_sq_min.pdf', bbox_inches='tight')
    
def q6():
    print(SEP, 'Q6', SEP)
    
    xs = np.linspace(1, 10, 10)
    ys = np.array((51, 103, 150, 199, 251, 303, 347, 398, 452, 512), dtype='f')
    errs = np.array((1, 1, 2, 2, 3, 3, 4, 5, 6, 7), dtype='f')

    m, c, alpha_m, alpha_c = calcfit(xs, ys, errs)
    
    print('Errors as given in question')
    print("""Best-fit gradient: {:.3g}
Best-fit intercept: {:.1g}
Error in gradient: {:.1g}
Error in intercept: {:.1g}"""
          .format(m, c, alpha_m, alpha_c))

    errs = np.full_like(errs, 4.0)
    m, c, alpha_m, alpha_c = calcfit(xs, ys, errs)
    
    print('Errors of 4 for every datum')
    print("""Best-fit gradient: {:.3g}
Best-fit intercept: {:.1g}
Error in gradient: {:.1g}
Error in intercept: {:.1g}"""
          .format(m, c, alpha_m, alpha_c))

    errs = np.full_like(errs, 8.0)
    errs[0] = errs[-1] = 1.0

    m, c, alpha_m, alpha_c = calcfit(xs, ys, errs)
    
    print('Better errors for first and last data')
    print("""Best-fit gradient: {:.3g}
Best-fit intercept: {:.1g}
Error in gradient: {:.1g}
Error in intercept: {:.1g}"""
          .format(m, c, alpha_m, alpha_c))
    
if __name__ == '__main__':
    q2()
    q3()
    q4and5()
    q6()
