import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner


def linear_model(epoch, t0, P):
    t = t0 + P * epoch
    return t


def multi_harmonic_model(epoch, tmid, t0, P, A_pair, P_pair, t0_pair):
    """transit mid point model including sinusoidal term"""
    # linear term
    t = t0 + P * epoch

    # sinusoidal perturbation
    ttv = A_pair * np.sin(2 * (np.pi / P_pair) * (tmid - t0_pair))

    return t + ttv


def log_posterior_prob(walker_position, resonances, data):
    """_summary_

    Args:
        walker_position (array-like): array containing model parameters
        resonances (tuple): tuple specifying planets in resonance
        data (array-like): data should have the structure. Length = num of planets in resonance. Each index contains a list of epoch, tmid, tmid_unc. Each index has dimensions 3 x number-of-measurements.
    """
    num_planets = len(data)
    total_likelihood = 0.0

    for i in range(num_planets):
        # compute likelihood for each planet and sum over the number of planets
        total_likelihood += single_planet_log_posterior(
            walker_position, resonances, data, planet_id=i
        )  # planet_id corresponds to planet index in data. Planets labeled sequentially when specified.
        return total_likelihood


def single_planet_log_posterior(walker_position, resonances, data, planet_id):
    # pull data
    epoch = data[:, 0]
    tmid = data[:, 1]
    tmid_unc = data[:, 2]

    # zero point
    t0 = walker_position[2 * planet_id]
    if (
        0.0 < t0 < 100000.0
    ):  # uniform log priors # TODO restrict to 10% of nominal value
        t0_prior = 0.0
    else:
        t0_prior = -np.inf  # outside range it is zero (recall this is a log prior)

    # linear period
    P = walker_position[2 * planet_id + 1]
    if 0.0 < P < 100.0:
        P_prior = 0.0
    else:
        P_prior = -np.inf

    # combine priors from linear single planet term
    log_prior = t0_prior + P_prior

    num_planets = len(planet_id)
    for r in range((len(resonances))):
        if planet_id in resonances[r]:

            # resonant phase
            t0_r = walker_position[2 * num_planets + 4 * r]
            # Set t0_pair to fraction of period. As long as the window size is not larger than period.
            if 1000 < t0_r < 10000:
                t0_r_prior = 0.0
            else:
                t0_r_prior = -np.inf

            # resonant period
            P_r = walker_position[2 * num_planets + 4 * r + 1]
            if 1000.0 < P_r < 2000.0:
                P_r_prior = 0.0
            else:
                P_r_prior = -np.inf

                # resonant amplitude
                A_r = walker_position[
                    2 * num_planets
                    + 4 * r
                    + (2 if resonances[r][0] == planet_id else 3)
                ]
            if 0 < A_r < 0.3:
                A_r_prior = 0.0
            else:
                A_r_prior = -np.inf

                # combine priors from resonant interaction
                log_prior += t0_r_prior + P_r_prior + A_r_prior

        single_planet_parameters = (
            t0,
            P,
        )
        tmid_pred = multi_harmonic_model(
            epoch, tmid, *single_planet_parameters
        )  # predicted tmid at the tmid observed

        log_p_likelihood = -0.5 * np.sum(
            (tmid - tmid_pred) ** 2 / tmid_unc**2 + np.log(tmid_unc**2)
        )  # this is the log likelihood function

        log_posterior_probability = log_p_likelihood + log_prior

        return log_posterior_probability


def log_posterior_prob(walker_position, planet, epoch, tmid, tmid_unc):
    """posterior probability of model tmid"""

    # sampled parameters i.e. where the walker is at currently
    t0 = walker_position[0]
    P = walker_position[1]
    A_pair = walker_position[2]
    P_pair = walker_position[3]
    t0_pair = walker_position[4]

    tmid_pred = multi_harmonic_model(
        epoch, tmid, *walker_position
    )  # predicted tmid at the tmid observed

    # STEP ONE: likelihood function. Lets work in log space to avoid numerical issues
    log_p_likelihood = -0.5 * np.sum(
        (tmid - tmid_pred) ** 2 / tmid_unc**2 + np.log(tmid_unc**2)
    )  # this is the log likelihood function

    # STEP TWO: Decide priors. log normal pdf around lmfit parameters
    if 0.0 < t0 < 100000.0:  # uniform log priors # restrict to 10% of nominal value
        t0_prior = 0.0
    else:
        t0_prior = -np.inf  # outside range it is zero (recall this is a log prior)

    if 0.0 < P < 100.0:
        P_prior = 0.0
    else:
        P_prior = -np.inf

    if 0 < A_pair < 0.3:
        A_pair_prior = 0.0
    else:
        A_pair_prior = -np.inf

    if 1000.0 < P_pair < 2000.0:
        P_pair_prior = 0.0
    else:
        P_pair_prior = -np.inf

    # Set t0_pair to fraction of period. As long as the window size is not larger than period.
    if (1000.0 < P_pair < 2000.0) and (P_pair * 2 / 3 < t0_pair < P_pair * 4 / 3):
        t0_pair_prior = 0.0
    else:
        t0_pair_prior = -np.inf

    log_prior = t0_prior + P_prior + A_pair_prior + P_pair_prior + t0_pair_prior

    if not np.isfinite(log_p_likelihood):  # make sure probabilities are finite
        log_p_likelihood = -np.inf

    if not np.isfinite(log_prior):
        log_prior = -np.inf

    # STEP THREE: Compute posterior (product of likelihood and prior)
    log_posterior_probability = log_p_likelihood + log_prior

    return log_posterior_probability


def emcee_sampler(data, nwalker, parameters, planet, nsteps, burnin):

    ndim = len(parameters)
    t0_init = np.random.uniform(2000, 2500, size=nwalker)
    P_init = np.random.uniform(5, 15, size=nwalker)
    A_pair_init = np.random.uniform(0, 0.5, size=nwalker)
    P_pair_init = np.random.uniform(1500, 2000, size=nwalker)
    t0_pair_init = np.random.uniform(1000, 2000, size=nwalker)

    p0 = np.array([t0_init, P_init, A_pair_init, P_pair_init, t0_pair_init]).T

    epoch = planet.epoch
    tmid = planet.tmid
    tmid_unc = planet.tmid_unc

    print(f"Launching {nwalker} walkers...\n")
    sampler = emcee.EnsembleSampler(
        nwalker,
        ndim,
        log_posterior_prob,
        args=(planet, epoch, tmid, tmid_unc),
    )
    state = sampler.run_mcmc(p0, burnin, progress=True)  # burn in
    sampler.reset()
    print(f"\nBunrin for {burnin} steps complete!\n")
    sampler.run_mcmc(state, nsteps, progress=True)  # main mcmc chain
    print(f"\nMCMC walk for {nsteps} steps complete!\n")

    # plot walkers
    fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(parameters[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.tight_layout()
    plt.savefig(dpi=200, fname=f"./figures/walkers_{planet.name}.pdf")

    # how long until chain "forgets" where it started. Use to refine burnin
    # tau = sampler.get_autocorr_time(quiet=True)
    # print(tau)

    # plot corver plot
    flat_samples = sampler.get_chain(
        discard=100, thin=15, flat=True
    )  # flatten chain to plot
    fig = corner.corner(
        flat_samples,
        labels=parameters,
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    fig.savefig(dpi=200, fname=f"./figures/corners_{planet.name}.pdf")

    # plot model
    fig, ax = plt.subplots(figsize=(8, 6))
    epoch_plot = np.linspace(min(planet.epoch), max(planet.epoch), 400)
    tmid_plot = np.linspace(min(planet.tmid), max(planet.tmid), 400)
    inds = np.random.randint(len(flat_samples), size=100)

    for ind in inds:

        sample = flat_samples[ind]
        t0, P, A_pair, P_pair, t0_pair = sample

        model_plot = multi_harmonic_model(
            epoch_plot, tmid_plot, t0, P, A_pair, P_pair, t0_pair
        )
        ttv_plot = model_plot - tmid_plot

        ax.plot(  # plot a few of the mcmc runs
            tmid_plot,
            ttv_plot,
            alpha=0.1,
        )

    ax.set_xlabel("BKJD (days)")
    ax.set_ylabel("TTVs (days)")
    ax.set_title(f"TTVs of planet {planet.name}")

    # stats
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(mcmc[1], q[0], q[1], parameters[i])

    # update planet parameters
    a = 1  # percentile index 0:16, 1:50, 2:84

    planet.t0 = np.percentile(flat_samples[:, 0], [16, 50, 84])[a]
    # planet.t0 = np.median(flat_samples[:, 0])
    planet.t0_unc_upp = np.diff(np.percentile(flat_samples[:, 0], [16, 50, 84]))[0]
    planet.t0_unc_low = np.diff(np.percentile(flat_samples[:, 0], [16, 50, 84]))[1]

    planet.P = np.percentile(flat_samples[:, 1], [16, 50, 84])[a]
    planet.P_unc_upp = np.diff(np.percentile(flat_samples[:, 1], [16, 50, 84]))[0]
    planet.P_unc_low = np.diff(np.percentile(flat_samples[:, 1], [16, 50, 84]))[1]

    planet.A_pair = np.percentile(flat_samples[:, 2], [16, 50, 84])[a]
    planet.A_pair_upp = np.diff(np.percentile(flat_samples[:, 2], [16, 50, 84]))[0]
    planet.A_pair_low = np.diff(np.percentile(flat_samples[:, 2], [16, 50, 84]))[1]

    planet.P_pair = np.percentile(flat_samples[:, 3], [16, 50, 84])[a]
    planet.P_pair_unc_upp = np.diff(np.percentile(flat_samples[:, 3], [16, 50, 84]))[0]
    planet.P_pair_unc_low = np.diff(np.percentile(flat_samples[:, 3], [16, 50, 84]))[1]

    planet.t0_pair = np.percentile(flat_samples[:, 4], [16, 50, 84])[a]
    planet.t0_pair_unc_upp = np.diff(np.percentile(flat_samples[:, 4], [16, 50, 84]))[0]
    planet.t0_pair_unc_low = np.diff(np.percentile(flat_samples[:, 4], [16, 50, 84]))[1]

    # ax.errorbar(  # plot linear term from mcmc run
    #    planet.tmid,
    #    planet.tmid - (planet.t0 + planet.epoch * planet.P),
    #    yerr=planet.tmid_unc,
    #    zorder=3,
    #    fmt=".",
    #    color="black",
    #    label="MCMC linear term",
    # )

    # ax.errorbar(  # plot linear term from mcmc run
    #    planet.tmid,
    #    planet.tmid
    #    - multi_harmonic_model(
    #        planet.epoch,
    #        planet.tmid,
    #        planet.t0,
    #        planet.P,
    #        planet.A_pair,
    #        planet.P_pair,
    #        planet.t0_pair,
    #    ),
    #    yerr=planet.tmid_unc,
    #    zorder=3,
    #    fmt=".",
    #    color="black",
    #    label="MCMC linear term",
    # )

    plt.legend()

    plt.savefig(dpi=200, fname=f"./figures/model_{planet.name}.pdf")
