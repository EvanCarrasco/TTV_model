from TTV_sin_model import System

planet_params_path = "./data/V1298_Tau_planet_properties.txt"
planet_obs_path = "./data/V1298_Tau_Trans.txt"

# create system object
system = System("V1298_TAU")

# populate system with planets
system.build_system(planet_params_path, planet_obs_path)

# fit linear model. Will use these coeffs as priors.
system.fit_linear_model(verbose=False, plot=False)

# specify which parameters to model
parameters = ["t0", "P", "A_pair", "P_pair", "t0_pair"]

# specify which planets we want to model
planets_to_model = ("c", "e", "d", "b")

# run MCMC model
# system.model_TTV(
#    parameters, planets_to_model, N_walkers=100, n_steps=10000, burnin=5000
# )

# plot from saved chain
system.plot_TTV(parameters, planets_to_model)
