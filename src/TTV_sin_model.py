import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model
from TTV_sampler_emcee import emcee_sampler, plot_mcmc


class Planet:
    def __init__(
        self,
        name,
        P,
        P_unc_upp,
        P_unc_low,
        epoch,
        tmid,
        tmid_unc,
    ):
        """
        Class containing planet parameters and TTV data. We will use this to calculate and store individual planet parameters.

        Parameters
        ----------
        name : str
            Planet name or label (e.g. 'b', 'c', etc.)

        Rp_Pstar : float

        b :float

        P : float
            Nominal orbital period [days] from Livingston et al.

        T0 : float
            Referance transit time [BKJD] from Livingston et al.

        T14 : float

        epoch : array-like
            Integer epoch numbers

        tmid : array-like
            Observed transit mid-times corresponding to each epoch [BKJD]

        """
        self.name = name

        # self.Rp_Rstar = Rp_Rstar
        # self.Rp_Rstar_unc_upp = Rp_Rstar_unc_upp
        # self.Rp_Rstar_unc_low = Rp_Rstar_unc_low

        # self.b = b
        # self.b_unc_upp = b_unc_upp
        # self.b_unc_low = b_unc_low

        self.P = P
        self.P_unc_upp = P_unc_upp
        self.P_unc_low = P_unc_low

        self.t0 = 0.0
        self.t0_unc_upp = 0.0
        self.t0_unc_low = 0.0

        # self.t14 = t14
        # self.t14_unc_upp = t14_unc_upp
        # self.t14_unc_upp = t14_unc_low

        self.tmid = tmid
        self.tmid_unc = tmid_unc

        self.epoch = epoch

        self.P_linear = 0.0
        self.P_linear_unc = 0.0

        self.t0_linear = 0.0
        self.t0_linear_unc = 0.0

        self.A_pair = 0.0
        self.A_pair_unc = 0.0

        self.P_pair = 0.0
        self.P_pair_unc = 0.0

        self.t0_pair = 0.0
        self.t0_pair_unc = 0.0

        self.linear_res = []

    def linear_transit_time(self, epoch):
        """Linear transit time model

        Args:
            epoch (array-like): epoch data

        Returns:
            array like: linear transit time
        """
        return self.t0 + self.P * epoch

    def linear_residuals(self):
        """compute TTV residual tmid is the transit midpoint"""
        return self.tmid - self.linear_transit_time(epoch=self.epoch)

    def plot_linear_transit_midpoint(self, ax=None, show=False):
        """plot observed and predicted transit midpoints as a function of epoch"""

        t_plot = np.linspace(min(self.tmid), max(self.tmid), 100)
        epoch_plot = np.linspace(min(self.epoch), max(self.epoch), 100)

        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(
            self.tmid,
            self.linear_transit_time(epoch=self.epoch),
            color="red",
            s=5,
            zorder=10,
        )
        ax.scatter(
            self.linear_transit_time(epoch=self.epoch),
            self.linear_transit_time(epoch=self.epoch),
            color="blue",
            s=5,
            marker="x",
            zorder=10,
        )
        ax.axvline(self.t0)
        ax.plot(t_plot, self.linear_transit_time(epoch=epoch_plot))

        ax.set_xlabel("BKJD (days)")
        ax.set_ylabel("predicted transit midpoint")
        ax.set_title("Test")

        if show:
            plt.show()
        return ax

    def plot_ttvs(self, ax=None, show=False):
        """plot TTV for a single planet"""
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.tmid, self.linear_residuals())
        ax.set_xlabel("BKJD (days)")
        ax.set_ylabel("TTVs (days)")
        ax.set_title(f"TTVs of planet {self.name}")
        if show:
            plt.show()
        return ax

    def scipy_linear_transit_times(self, verbose=False, plot=False):
        def linear_transit_model(epoch, t0, period):
            return t0 + epoch * period

        popt, pcov = curve_fit(linear_transit_model, self.epoch, self.tmid)
        perr = np.sqrt(np.diag(pcov))

        P = popt[0]
        P_unc = perr[0]

        t0 = popt[1]
        t0_unc = perr[1]

        self.P_linear = P
        self.P_linear_unc = P_unc

        self.t0_linear = t0
        self.t0_linear_unc = t0_unc

        self.linear_res = self.tmid - (self.t0_linear + self.P_linear_unc * self.epoch)

        if verbose:
            print(
                f"scipy.optimize fit for planet {self.name}\n------------------------------------------------------------------------"
            )
            print(f"Period: {self.P_linear:.8f} +/- {self.P_linear_unc:.8f}")
            print(f"t0: {self.t0_linear:.8f} +/- {self.t0_linear_unc:.8f}\n")

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(
                self.tmid,
                self.tmid - linear_transit_model(self.epoch, self.t0, self.tmid),
            )
            plt.show()

    def lmfit_linear_transit_times(self, verbose=False, plot=False):
        def linear_transit_model(epoch, t0, period):
            return t0 + epoch * period

        orbit_model = Model(linear_transit_model)
        result = orbit_model.fit(
            self.tmid,
            epoch=self.epoch,
            t0=self.t0,
            period=(self.tmid[0] - self.tmid[1]),
            weights=1,
            method="leastsq",
            # weights=1 / self.tmid_unc,
        )
        # REMEMBER ME !!! ONCE YOU GET THIS TO WORK THIS IS WHERE YOU WILL ASSIGN THE T0 AND P PARAMETERS !!!!!

        # assign t0 and P from linear fit to planet object
        self.t0_linear = result.params["t0"].value
        self.t0_linear_unc = result.params["t0"].stderr

        self.P_linear = result.params["period"].value
        self.P_linear_unc = result.params["period"].stderr

        self.linear_res = self.tmid - (self.t0_linear + self.P_linear * self.epoch)

        if verbose:
            print(
                f"\nlm fit for planet {self.name}\n------------------------------------------------------------------------\n"
            )
            print(result.fit_report())

        if plot:
            plt.scatter(self.tmid, self.linear_res)
            plt.show()

        return result

    def lmfit_sinusodial_transit_time(self, verbose=False, plot=False):  # not working
        t0 = self.t0
        P = self.P

        def sinusodial_transit_model(
            t,
            A_pair,
            P_pair,
            t0_pair,
        ):
            return A_pair * np.sin((2 * np.pi) / P_pair * (t - t0_pair))

        orbit_model = Model(sinusodial_transit_model)
        orbit_model.set_param_hint("A_pair", min=0.05, max=0.1)
        orbit_model.set_param_hint("P_pair", min=1500, max=2000)

        result = orbit_model.fit(
            self.linear_res,
            t=self.tmid,
            A_pair=0.001,
            P_pair=100.0,
            t0_pair=0.0,
            weights=1 / self.tmid_unc,
            method="leastsq",  # can change to "leastsq" (lmfit Levenberg–Marquardt) or "least_squares" (scipy least squares fitting)
            # loss="soft_l1",  # downweights outliers
        )

        # assign fitting parameters planet object. This will be used as priors for MCMC sampler
        # self.t0 = result.params["t0"].value
        # self.t0_unc_upp = result.params["t0"].stderr

        # self.P = result.params["period"].value
        # self.P_unc_upp = result.params["period"].stderr

        self.A_pair = result.params["A_pair"].value
        self.A_pair_unc = result.params["A_pair"].stderr

        self.P_pair = result.params["P_pair"].value
        self.P_pair_unc = result.params["P_pair"].stderr

        self.t0_pair = result.params["t0_pair"].value
        self.t0_pair_unc = result.params["t0_pair"].stderr

        if verbose:
            print(
                f"\nlmfit sinusodial for planet {self.name}\n------------------------------------------------------------------------\n"
            )
            print(result.fit_report())
            print("\n")
            print(f"convergance: {result.success}", result.message)
            # print(result.covar)  # covariance matrix, should not be singular

        if plot:
            epoch_plot = np.linspace(min(self.epoch), max(self.epoch), 400)
            tmid_plot = np.linspace(min(self.tmid), max(self.tmid), 400)

            model_tmid = sinusodial_transit_model(
                tmid_plot, self.A_pair, self.P_pair, self.t0_pair
            )

            # model_ttv = (self.t0_linear + self.P_linear * epoch_plot) - model_tmid

            plt.plot(tmid_plot, model_tmid)

            plt.errorbar(
                self.tmid,
                self.linear_res,
                yerr=self.tmid_unc,
                zorder=3,
                fmt=".",
                color="black",
            )

            plt.show()

        return result


class System:
    def __init__(self, name, planets=None):
        """
        Class containing all planets in a system. We will use this to model interactions between planets.

        parameters
        ----------
        name : str
            System name or label (e.g. V1298_TAU)

        planets : array-like
            List of planet objects in a system
        """

        self.name = name
        self.planets = planets if planets is not None else []

    def build_system(self, planet_params_path, planet_obs_path):
        """create a planet object for each planet in the planet params file and populate the object with params and transit data from the data file"""
        planet_names = []

        # first we will create some planet objects that hold the planet parameters given in the params file
        try:
            with open(planet_params_path, "r") as file:
                # pull out planet names from params file
                next(file)  # Skips the first line
                for line in file:
                    # columns should correspond to #planet P P_unc_upp P_unc_low
                    columns = line.split(" ")
                    # check if this line contians a new planet
                    if len(columns) >= 2 and columns[0] not in planet_names:
                        planet_names.append(
                            columns[0]
                        )  # create planet object from unique planets in params file
                        self.planets.append(
                            Planet(
                                name=columns[0],
                                P=float(columns[1]),
                                P_unc_upp=float(columns[2]),
                                P_unc_low=float(columns[3]),
                                epoch=[],
                                tmid=[],
                                tmid_unc=[],
                            )
                        )
                    elif len(columns) == 1 and columns[0] not in planet_names:
                        planet_names.append(columns[0])
                        self.planets.append(
                            Planet(
                                name=columns[0],
                                epoch=[],
                                tmid=[],
                                tmid_unc=[],
                            )
                        )
                    else:
                        print("Params file malformed")
        except ValueError:
            print("File does not exist...")

        try:
            with open(planet_obs_path, "r") as file:
                # now lets add all of the juicy TTV data to the planet objects we just created
                for p in self.planets:
                    epoch_list = []
                    tmid_list = []
                    tmid_unc_list = []
                    for line in file:
                        columns = line.split(" ")
                        if columns[0] == p.name:
                            epoch_list.append(int(columns[1]))
                            tmid_list.append(float(columns[2]))
                            tmid_unc_list.append(float(columns[3]))

                    p.epoch = np.array(epoch_list)
                    p.tmid = np.array(tmid_list)
                    p.tmid_unc = np.array(tmid_unc_list)
                    file.seek(0)

        except ValueError:
            print("File does not exist...")

        print(
            f"System {self.name} constructed! \nPlanet identified: {str(planet_names)}."
        )

        # build session
        try:
            session = f"./sessions/{self.name}"
            if not os.path.exists(session):
                os.makedirs(session)
                os.makedirs(session + "/figures")
                os.makedirs(session + "/figures/model")
                os.makedirs(session + "/figures/corner")
                os.makedirs(session + "/figures/walker")
                os.makedirs(session + "/mcmc_run")
            # for (
            #    root,
            #    dirs,
            #    files,
            # ) in os.walk(
            #    session + "/figures"
            # ):  # remove any figures in session figures file to prevent overwrite conflicts
            #    for file in files:
            #        if file.endswith(".pdf"):
            #            os.remove(os.path.join(root, file))
            print("Session setup complete.\n")
        except:
            print("Session setup bad.\n")
        return

    def plot_all_linear_transit_midpoint(self, plot=True, save=False):
        """plot all observed and predicted transit mid points as a function of epoch"""
        n = len(self.planets)
        fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n), sharex=False)
        if n == 1:
            axes = [axes]
        for ax, p in zip(axes, self.planets):
            p.plot_linear_transit_midpoint(ax=ax)
            ax.set_title(p.name)
        fig.suptitle(f"{self.name} system transit mid point")
        plt.tight_layout()

        if save:
            plt.savefig(fname="all_transit_midpoint.pdf", dpi=200)
        if plot == True:
            plt.show()
        return

    def plot_all_ttv(self, plot=True, save=False):
        """plot TTVs for all of the planets in the system"""
        n = len(self.planets)
        fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, p in zip(axes, self.planets):
            p.plot_ttvs(ax=ax)
            ax.set_title(p.name)
        fig.suptitle(f"{self.name} system TTVs")
        plt.tight_layout()

        if save:
            plt.savefig(fname="all_TTVs.pdf", dpi=200)
        if plot == True:
            plt.show()
        return

    def model_TTV(self, parameters, planets_to_model, N_walkers, n_steps, burnin):
        """select pair of planets and run MCMC model to find best fit parameters"""

        planets = (
            []
        )  # list to contain the planets we want modeled. This will be a list of planet objects corresponding to all of the planets we want modeled in our system.

        for planet in planets_to_model:
            for bodies in self.planets:
                if planet == bodies.name:
                    planets.append(bodies)
        print("\n\nyou have selected the following planets to model:")
        for planet in planets:
            print(planet.name)
        print("\n")

        for p in planets:
            emcee_sampler(N_walkers, parameters, p, n_steps, burnin, self.name)
        return

    def plot_TTV(self, parameters, planets_to_model):
        planets = []

        for planet in planets_to_model:
            for bodies in self.planets:
                if planet == bodies.name:
                    planets.append(bodies)
        for p in planets:
            plot_mcmc(p, parameters, self.name)
        return

    # def model_TTV_entire_system(self, resonances, N_walkers, n_steps, burnin):
    #    data = []  # initialize data structure

    #    # rebuild data structure to be amendable to user selected resonant pairs
    #    # should have shape 3 x len(obs) x n where n is the number of planets and obs is the data.
    #    # this is an inhomogenous list since each planet has a diffrent data shape. Be careful.
    #    for planet in self.planets:  # for each planet in the system object
    #        data.append(planet.epoch)
    #        data.append(planet.tmid)
    #        data.append(planet.tmid_unc)

    #    # pass restructured data into sampler
    #    emcee_sampler(data, resonances, N_walkers nsteps, burnin)
    #    return

    def fit_linear_model(self, verbose, plot):
        """Fir linear model to TTV for all planets in a system.
        Args:
            verbose (bool, optional): Print fitting results. Defaults to False.
            plot (bool, optional): Plot fitting results. Defaults to False.
        """
        for p in self.planets:
            p.lmfit_linear_transit_times(verbose=verbose, plot=plot)

    def fit_sinusodial_model(self, verbose, plot):
        """Fit sinusodial model to TTV for all planets in a system.

        Args:
            verbose (bool, optional):  Print fitting results. Defaults to False.
            plot (bool, optional): Plot fitting results. Defaults to False.
        """
        for p in self.planets:
            p.lmfit_sinusodial_transit_time(verbose=verbose, plot=plot)
