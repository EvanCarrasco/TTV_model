import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
import scipy.stats


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


def log_posterior_prob(walker_position, epoch, tmid, tmid_unc):
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
    if 0.0 < t0 < 3000.0:  # uniform log priors # restrict to 10% of nominal value
        t0_prior = 0.0
    else:
        t0_prior = -np.inf  # outside range it is zero (recall this is a log prior)

    if 0.0 < P < 100.0:
        P_prior = 0.0
    else:
        P_prior = -np.inf

    if 0 < A_pair < 0.5:
        A_pair_prior = 0.0
    else:
        A_pair_prior = -np.inf

    if 500.0 < P_pair < 5000.0:
        P_pair_prior = 0.0
    else:
        P_pair_prior = -np.inf

    # convert t0_pair to a phase [=] radians
    if 0 < 2 * np.pi * t0_pair / P_pair < 2 * np.pi:
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


def emcee_sampler(nwalker, parameters, planet, nsteps, burnin, session_name):
    # setup backend to save chain
    backend = emcee.backends.HDFBackend(
        f"sessions/{session_name}/mcmc_run/{session_name}_{planet.name}_chain.h5"
    )

    # create starting pos
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

    # launch walkers
    print(f"Launching {nwalker} walkers...\n")
    sampler = emcee.EnsembleSampler(
        nwalker, ndim, log_posterior_prob, args=(epoch, tmid, tmid_unc), backend=backend
    )
    state = sampler.run_mcmc(p0, burnin, progress=True)  # burn in
    sampler.reset()
    print(f"\nBunrin for {burnin} steps complete!\n")
    sampler.run_mcmc(state, nsteps, progress=True)  # main mcmc chain
    print(f"\nMCMC walk for {nsteps} steps complete!\n")

    # how long until chain "forgets" where it started. Use to refine burnin
    # tau = sampler.get_autocorr_time(quiet=True)
    # print(tau)
    return


def plot_mcmc(planet, parameters, session_name):
    backend = emcee.backends.HDFBackend(
        f"sessions/{session_name}/mcmc_run/{session_name}_{planet.name}_chain.h5"
    )
    session_figures_path = f"./sessions/{session_name}/figures"

    samples = backend.get_chain()
    ndim = len(parameters)

    # plot walkers
    fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(parameters[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.tight_layout()
    plt.savefig(
        dpi=200, fname=f"{session_figures_path}/walker/walkers_{planet.name}.pdf"
    )

    # plot corver plot
    flat_samples = backend.get_chain(flat=True)  # flatten chain to plot
    fig = corner.corner(
        flat_samples,
        labels=parameters,
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    fig.savefig(
        dpi=200, fname=f"{session_figures_path}/corner/corners_{planet.name}.pdf"
    )

    # stats
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(mcmc[1], q[0], q[1], parameters[i], planet.name)

    # update planet parameters
    a = 1  # percentile index 0:16, 1:50, 2:84

    planet.t0 = np.percentile(flat_samples[:, 0], [16, 50, 84])[a]
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

    median_model = multi_harmonic_model(
        planet.epoch,
        planet.tmid,
        planet.t0,
        planet.P,
        planet.A_pair,
        planet.P_pair,
        planet.t0_pair,
    )

    chi2 = np.sum((planet.tmid - median_model) ** 2 / planet.tmid_unc**2)

    red_chi2 = chi2 / (len(planet.tmid) - ndim)

    print(red_chi2, planet.name)

    # plot model
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    axes = axes.flatten()

    epoch_plot = np.linspace(min(planet.epoch), max(planet.epoch), 400)
    tmid_plot = np.linspace(min(planet.tmid), max(planet.tmid), 400)
    inds = np.random.randint(len(flat_samples), size=1000)

    ax = axes[0]
    for ind in range(len(inds)):

        sample = flat_samples[ind]
        t0, P, A_pair, P_pair, t0_pair = sample

        ttv_plot = A_pair * np.sin(2 * np.pi / P_pair * (tmid_plot - t0_pair))

        ax.plot(  # plot a few of the mcmc runs
            tmid_plot,
            ttv_plot,
            alpha=0.1,
        )

    median_model_continous = planet.A_pair * np.sin(
        2 * np.pi / planet.P_pair * (tmid_plot - planet.t0_pair)
    )

    ax.plot(
        tmid_plot, median_model_continous, color="black"
    )  # plot continous median model

    ax.set_xlabel("BKJD (days)")
    ax.set_ylabel("TTVs (days)")
    ax.set_title(f"TTVs of planet {planet.name}")

    ax.errorbar(  # plot linear term from mcmc run
        planet.tmid,
        planet.tmid - (planet.t0 + planet.epoch * planet.P),
        yerr=planet.tmid_unc,
        zorder=3,
        fmt=".",
        color="black",
        label="MCMC linear term",
    )

    ax = axes[1]
    ax.errorbar(  # plot model residual
        planet.tmid,
        planet.tmid - median_model,
        yerr=planet.tmid_unc,
        zorder=3,
        fmt=".",
        color="black",
    )
    ax.set_ylabel("Model residual")

    plt.tight_layout()
    plt.savefig(dpi=200, fname=f"{session_figures_path}/model/model_{planet.name}.pdf")
