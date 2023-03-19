# @File          :   simulator_2ss.py
# @Last modified :   2022/12/05 13:56:46
# @Author        :   Matthias Gueltig

import numpy as np
from typing import Optional


class Simulator(object):
    """
    Solves Advection-Diffusion equation including Two-Site sorption.
    """
    def __init__(self, d_e: float, n_e: float, rho_s: float, beta: float,
                 f: float, k_d: float, cw_0: float, t_max: float,
                 x_right: float, x_steps: int, t_steps: int, v: float,
                 a_k: float, alpha_l: float, s_k_0: float, sand: bool,
                 n_e_sand: Optional[float] = None,
                 x_start_soil: Optional[float] = None,
                 x_stop_soil: Optional[float] = None,
                 alpha_l_sand: Optional[float] = None,
                 v_e_sand: Optional[float] = None,
                 is_noisy:bool = False,
                 noise:float = 0):
        """Constructor method initializing the parameters.

        Args:
            d_e (float):    molecular diffusion coeff.                          [L^2/T]
            n_e (float):    effective porosity                                  [-]
            rho_s (float):  dry bulk density of soil                            [M/L^3]
            beta (float):   parameter for sorption isotherms                    [-]
            f (float):      coupling kinetic and instantaneous sorption         [-]
            k_d (float):    parameter of sorption isotherm                      [L^3/M]
            cw_0 (float):   initial dissolved concentration                     [M/L^3]
            t_max (float):  end time of simulation                              [T]
            x_right (float):length of the 1D simulation field
            x_steps (int):  number of spatial steps between 0 and x_right
            t_steps (int):  number of time steps
            v (float):      advective velocity                                  [L/T]
            a_k (float):    first order rate constant of eq. 17c Simunek et al. [1/T]
            alpha_l (float):longitudinal dispersion coefficient                 [L]
            s_k_0 (float):  init. mass conc. of kinetically sorbed PFOS         [M/M]
            sand (bool):    True if sand data exists                
            Optional:
            n_e_sand (float, optional): Porosity of sand [-]
            x_start_soil (float, optional): spatial index at which sand turns into soil
            x_stop_soil (float, optional): spatial index at which soil turns back to sand
            alpha_l_sand (float, optional): longitudinal dispersion coefficient in sand [L]
            v_e_sand (float, optional): advective velocity in sand [L/T]
        """

        # set class parameters
        self.n_e = n_e
        self.beta = beta
        self.f = f
        self.k_d = k_d
        self.cw_0 = cw_0
        self.s_k_0 = s_k_0
        self.t_max = t_max
        self.x_right = x_right
        self.x_steps = x_steps
        self.a_k = a_k
        self.t_steps = t_steps
        self.rho_s = rho_s * (1-n_e) #TODO: ADDED FROM ME
        self.sand = sand


        #TODO: Noises
        #Noise is a shifted uniform
        self.is_noisy = is_noisy
        self.noises = []
        self.noise_factor = noise

        if self.sand:
            self.n_e_sand = n_e_sand
            self.x_start = x_start_soil
            self.x_stop = x_stop_soil

            self.v = np.ndarray(self.x_steps)
            
            self.v[:self.x_start] = v_e_sand
            self.v[self.x_start:self.x_stop] = v
            self.v[self.x_stop:] = v_e_sand

            # no molecular diffusion in sand
            disp_sand = v_e_sand*alpha_l_sand
            disp_soil = v*alpha_l+d_e

            self.disp = np.ndarray(self.x_steps)

            self.disp[:self.x_start] = disp_sand
            self.disp[self.x_start:self.x_stop] = disp_soil
            self.disp[self.x_stop:] = disp_sand

        else:
            self.v = np.ones(self.x_steps)*v
            self.disp = self.v*alpha_l+d_e

        # consider x_right cells
        self.x = np.linspace(0, self.x_right, self.x_steps)
        self.dx = self.x[1] - self.x[0]

        self.t = np.linspace(0, self.t_max, self.t_steps)
        self.dt = self.t[1] - self.t[0]

        # Use Freundlich isotherm for instantaneous sorption
        self.sorpt_isotherm = self.freundlich
        self.sorpt_derivat = self.d_freundlich

    def freundlich(self, c: np.ndarray):
        """implements freundlich sorpotion sorption isotherm with
        K_d [M/M], beta, c[M/L^3]"""
        return c**self.beta*self.k_d

    def d_freundlich(self, c: np.ndarray):
        """returns derivation(derivative) of freundlich sorption isotherm [M/TL^3]"""
        return c**(self.beta-1)*self.beta*self.k_d

    def generate_sample(self):
        """Function that generates solution for PDE problem.
        """

        #TODO: Added from Jonathan - Noises
        self.noise = np.random.rand()
        self.noises.append(self.noise)
        ##
        
        # Laplacian matrix for diffusion term
        nx = np.diag(-2*np.ones(self.x_steps), k=0)
        nx_minus_1 = np.diag(np.ones(self.x_steps-1), k=-1)
        nx_plus_1 = np.diag(np.ones(self.x_steps-1), k=1)
        self.lap = nx + nx_minus_1 + nx_plus_1
        self.lap /= self.dx**2

        # symmetric differences for advection term
        nx_fd = np.diag(np.ones(self.x_steps), k=0)
        nx_fd_plus_1 = np.diag(np.ones(self.x_steps-1)*(-1), k=-1)
        self.fd = nx_fd + nx_fd_plus_1
        self.fd /= self.dx

        # solution vector with c in first self.x_steps rows and s_k in last
        # self.x_step columns
        u = np.ndarray((2*self.x_steps, len(self.t)))
        u[:, 0] = np.zeros(2*self.x_steps)

        # add initial conditions
        if self.sand:
            u[self.x_start:self.x_stop, 0] = self.cw_0
            u[self.x_steps+self.x_start:self.x_steps+self.x_stop, 0] = \
                self.s_k_0
        else:
            u[:self.x_steps, 0] = self.cw_0
            u[self.x_steps:, 0] = self.s_k_0

        sol = self.solve_ivp_euler_exp(u=u)

        # sample_c: (x, t), dissolved conc.
        # sample_sk: (x, t), kin. sorbec conc.
        sample_c = sol[:self.x_steps, :]
        sample_sk = sol[self.x_steps:, :]

        return sample_c, sample_sk

    def solve_ivp_euler_exp(self, u):
        """Simple explicit Euler to integrate ode."""

        for i in range(len(self.t)-1):
            print(i)
            u[:, i+1] = u[:, i] + self.dt*self.ad_ode(t=i, conc_cw_sk=u[:, i])
        return u

    def ad_ode(self, t: np.ndarray, conc_cw_sk: np.ndarray):
        """Function that should be integrated over time.

        Args:
            t (np.ndarray): timestep
            conc_cw_sk (np.ndarray): stacked c_w and s_k
        """

        # split u in cw and sk
        cw = conc_cw_sk[:self.x_steps]
        sk = conc_cw_sk[self.x_steps:]

        # setups boundarys for cw which are not accesed by fd and lap
        # in case nothing else is needed put zeros in the array
        # top dirichlet boundary
        dif_bound = np.zeros(self.x_steps)
        dif_bound[0] = 0
        dif_bound[-1] = self.disp[-1]/(self.dx**2)*cw[-1]

        # sand scenario
        if self.sand:
            sk_soil = sk[self.x_start:self.x_stop]
            cw_soil = cw[self.x_start:self.x_stop]

            f_hyd = np.zeros(self.x_steps)
            g_hyd = np.zeros(self.x_steps)
            ret = np.ones(self.x_steps)

            # create inhomogenous and divisor values for soil area
            f_hyd[self.x_start:self.x_stop] = -self.a_k * \
                (1-self.f) * (self.k_d * cw_soil**(self.beta-1)) * \
                (self.rho_s/self.n_e)

            g_hyd[self.x_start:self.x_stop] = self.a_k * self.rho_s/self.n_e \
                * sk_soil
            
            ##########################################################
            #Here I want to add NOISE
            if self.is_noisy:
                noise = self.noise_factor/2 * cw_soil * (self.noise - 0.5) 
            else:
                noise = 0
            ret[self.x_start:self.x_stop] = 1/ (1 + self.f * \
                self.sorpt_derivat(cw_soil) * (self.rho_s/self.n_e)) + noise
            ##########################################################


            
            # calculate change of cw and sk over over time
            cw_new = (self.disp*np.matmul(self.lap, cw) + dif_bound -
                      self.v * np.matmul(self.fd, cw) + f_hyd * cw +
                      g_hyd) * ret

            sk_new = np.zeros(self.x_steps)
            sk_new[self.x_start:self.x_stop] = - \
                f_hyd[self.x_start:self.x_stop] * (self.n_e/self.rho_s) * \
                cw_soil - g_hyd[self.x_start:self.x_stop] * \
                (self.n_e/self.rho_s)
        # non sand scenario
        else:
            f_hyd = np.zeros(self.x_steps)
            g_hyd = np.zeros(self.x_steps)
            ret = np.ones(self.x_steps)

            f_hyd = -self.a_k * \
                (1-self.f)*(self.k_d*cw**(self.beta-1))*(self.rho_s/self.n_e)
            g_hyd = self.a_k * self.rho_s/self.n_e * sk
            
            ################################################################
            #TODO:   THIS IS IMPORTANT
            ret = (self.f*self.sorpt_derivat(cw)*(self.rho_s/self.n_e))+1
            ################################################################
            
            cw_new = (self.disp*np.matmul(self.lap, cw) + dif_bound)/ret
            -self.v/ret*np.matmul(self.fd, cw) + (f_hyd*cw+g_hyd)/ret

            sk_new = -f_hyd * (self.n_e/self.rho_s) * cw - g_hyd * \
                (self.n_e/self.rho_s)

        # stack 2 calculated rate of changes and prepare for integration
        conc_cw_sk_new = np.ndarray(self.x_steps*2)
        conc_cw_sk_new[:self.x_steps] = cw_new
        conc_cw_sk_new[self.x_steps:] = sk_new

        return conc_cw_sk_new

