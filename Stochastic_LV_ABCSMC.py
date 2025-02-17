import numpy as np
from scipy.integrate import odeint
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt
from stoch_lv import gillespie_lv


# Lotka-Volterra model deterministic
def lotka_volterra_model1(y, t, a, b):
    """Standard LV model with parameters a, b."""
    x, y_pred = y
    dxdt = a * x - x * y_pred
    dydt = b * x * y_pred - y_pred
    return [dxdt, dydt]


def simulate_model1(theta, t_obs, noise_std=0.5):
    a, b = theta
    y0 = [1.0, 0.5]
    sol = odeint(lotka_volterra_model1, y0, t_obs, args=(a, b), atol=1e-8, rtol=1e-8)
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol.flatten()


# Lotka-Volterra model stochastic
def simulate_model2(theta, t_obs, noise_std=0.5):
    """Stochastic LV model using Gillespie algorithm."""
    c1, c2, c3 = theta
    y0 = [1.0, 0.5]
    X0 = int(round(y0[0]))
    Y0 = int(round(y0[1]))
    T_max = max(t_obs)

    # Run Gillespie simulation
    t_sim, X_sim, Y_sim = gillespie_lv(X0, Y0, c1, c2, c3, T_max)

    # Interpolate results to t_obs
    if len(t_sim) == 0:
        X_interp = np.zeros_like(t_obs)
        Y_interp = np.zeros_like(t_obs)
    else:
        X_interp = np.interp(t_obs, t_sim, X_sim, left=X0, right=X_sim[-1])
        Y_interp = np.interp(t_obs, t_sim, Y_sim, left=Y0, right=Y_sim[-1])

    # Combine and add noise
    combined = np.column_stack((X_interp, Y_interp))
    noisy_combined = combined + np.random.normal(0, noise_std, combined.shape)
    return noisy_combined.flatten()


# Repressilator models
def repressilator_deterministic(y, t, alpha0, n, beta, alpha):
    m1, p1, m2, p2, m3, p3 = y
    dm1dt = -m1 + alpha / (1 + p3 ** n) + alpha0
    dp1dt = -beta * (p1 - m1)
    dm2dt = -m2 + alpha / (1 + p1 ** n) + alpha0
    dp2dt = -beta * (p2 - m2)
    dm3dt = -m3 + alpha / (1 + p2 ** n) + alpha0
    dp3dt = -beta * (p3 - m3)
    return [dm1dt, dp1dt, dm2dt, dp2dt, dm3dt, dp3dt]


def simulate_repressilator_deterministic(theta, t_obs, noise_std=0.5):
    alpha0, n, beta, alpha = theta
    y0 = [0, 2, 0, 1, 0, 3]
    sol = odeint(repressilator_deterministic, y0, t_obs, args=(alpha0, n, beta, alpha), atol=1e-8, rtol=1e-8)
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol.flatten()


def gillespie_repressilator(theta, t_obs, noise_std=0.5):
    alpha0, n, beta, alpha = theta
    y0 = [0, 2, 0, 1, 0, 3]
    sol = odeint(repressilator_deterministic, y0, t_obs, args=(alpha0, n, beta, alpha), atol=1e-8, rtol=1e-8)
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol.flatten()


def simulate_repressilator_stochastic(theta, t_obs, noise_std=0.5):
    return gillespie_repressilator(theta, t_obs, noise_std)


# Distance function remains unchanged
def distance(sim_data, observed_data):
    return np.sum((sim_data - observed_data) ** 2)

# ABC SMC Algorithm for Model Selection
def abc_smc_model_selection(N, T, epsilons, models, model_prior, observed_data, t_obs):
    particles = []
    weights = []

    # Population 0: Sample from priors
    current_particles = {m: [] for m in range(len(models))}
    current_weights = {m: [] for m in range(len(models))}
    num_particles = 0
    while num_particles < N:
        m = np.random.choice(len(models), p=model_prior)
        params = [prior.rvs() for prior in models[m]['priors']]
        sim_data = models[m]['simulate'](params, t_obs)
        # Ensure observed_data matches the shape of sim_data
        if len(observed_data[m]) != len(sim_data):
            raise ValueError(f"Observed data for model {m} must have shape {len(sim_data)}.")
        if distance(sim_data, observed_data[m]) <= epsilons[0]:
            current_particles[m].append(params)
            current_weights[m].append(1.0)
            num_particles += 1
    # Normalize weights
    for m in current_particles:
        if current_particles[m]:
            current_weights[m] = np.array(current_weights[m]) / np.sum(current_weights[m])
    particles.append(current_particles)
    weights.append(current_weights)

    # populations
    for t in range(1, T):
        print(f"Processing population {t}/{T-1}")
        new_particles = {m: [] for m in range(len(models))}
        new_weights = {m: [] for m in range(len(models))}
        num_accepted = 0
        while num_accepted < N:
            m = np.random.choice(len(models), p=model_prior)
            # Skip if model has no particles in previous population
            if m not in particles[t-1] or not particles[t-1][m]:
                continue
            # Sample from previous particles of model m
            idx = np.random.choice(len(particles[t-1][m]), p=weights[t-1][m])
            prev_params = particles[t-1][m][idx]
            # Perturb parameters with Gaussian kernel
            proposed_params = []
            for i in range(len(prev_params)):
                sigma = models[m]['kernel_sigma'][i]
                proposed = prev_params[i] + np.random.normal(0, sigma)
                proposed_params.append(proposed)
            # Check prior bounds
            prior_ok = True
            for i, prior in enumerate(models[m]['priors']):
                if not (prior.ppf(0) <= proposed_params[i] <= prior.ppf(1)):
                    prior_ok = False
                    break
            if not prior_ok:
                continue
            # Simulate data
            sim_data = models[m]['simulate'](proposed_params, t_obs)
            # Ensure observed_data matches the shape of sim_data
            if len(observed_data[m]) != len(sim_data):
                raise ValueError(f"Observed data for model {m} must have shape {len(sim_data)}.")
            if distance(sim_data, observed_data[m]) <= epsilons[t]:
                # Calculate weight
                prior_density = 1.0
                for i, prior in enumerate(models[m]['priors']):
                    prior_density *= prior.pdf(proposed_params[i])
                # Compute denominator
                denominator = 0.0
                for i_prev, prev_p in enumerate(particles[t-1][m]):
                    kernel_density = 1.0
                    for j in range(len(prev_p)):
                        kernel_density *= norm(prev_p[j], models[m]['kernel_sigma'][j]).pdf(proposed_params[j])
                    denominator += weights[t-1][m][i_prev] * kernel_density
                weight = prior_density / (denominator + 1e-12)
                new_particles[m].append(proposed_params)
                new_weights[m].append(weight)
                num_accepted += 1
        # Normalize weights
        for m in new_particles:
            if new_particles[m]:
                new_weights[m] = np.array(new_weights[m]) / np.sum(new_weights[m])
        particles.append(new_particles)
        weights.append(new_weights)
    return particles, weights

# Example usage
if __name__ == "__main__":
    # Define models
    models = [
        {   # Model 0: Deterministic LV
            'simulate': simulate_model1,
            'priors': [uniform(loc=0.5, scale=1.5), uniform(loc=0.5, scale=1.5)],
            'kernel_sigma': [0.1, 0.1]
        },
        {   # Model 1: Stochastic LV
            'simulate': simulate_model2,
            'priors': [uniform(loc=0.5, scale=1.5), uniform(loc=0.5, scale=1.5), uniform(loc=0.1, scale=0.3)],
            'kernel_sigma': [0.1, 0.1, 0.05]
        },
        {   # Model 2: Deterministic repressilator
            'simulate': simulate_repressilator_deterministic,
            'priors': [uniform(loc=0, scale=2), uniform(loc=1, scale=3), uniform(loc=4, scale=2), uniform(loc=800, scale=400)],
            'kernel_sigma': [0.5, 0.5, 0.5, 100]
        },
        {   # Model 3: Stochastic repressilator
            'simulate': simulate_repressilator_stochastic,
            'priors': [uniform(loc=0, scale=2), uniform(loc=1, scale=3), uniform(loc=4, scale=2), uniform(loc=800, scale=400)],
            'kernel_sigma': [0.5, 0.5, 0.5, 100]
        }
    ]
    model_prior = [0.25, 0.25, 0.25, 0.25]  # Uniform prior over models

    # Generate observed data for each model
    t_obs = np.linspace(0, 15, 8)
    observed_data = {
        0: simulate_model1((1.0, 1.0), t_obs, noise_std=0.5),  # Model 0: LV with a=1.0, b=1.0
        1: simulate_model2((1.0, 1.0, 0.2), t_obs, noise_std=0.5),  # Model 1: LV with c1=1.0, c2=1.0, c3=0.2
        2: simulate_repressilator_deterministic((1, 2, 5, 1000), t_obs, noise_std=0.5),  # Model 2: Repressilator
        3: simulate_repressilator_stochastic((1, 2, 5, 1000), t_obs, noise_std=0.5)  # Model 3: Stochastic repressilator
    }

    # ABC SMC parameters
    N = 1000  # Number of particles
    T = 5     # Number of populations
    epsilons = [50.0, 30.0, 20.0, 10.0, 5.0]

    # Run ABC SMC for model selection
    particles, weights = abc_smc_model_selection(N, T, epsilons, models, model_prior, observed_data, t_obs)

    # results
    final_particles = particles[-1]
    final_weights = weights[-1]
    model_counts = {m: len(final_particles.get(m, [])) for m in range(len(models))}
    total = sum(model_counts.values())
    print("\nModel posterior probabilities:")
    for m in range(len(models)):
        print(f"Model {m}: {model_counts[m]/total:.2%}")

    # Plotting parameter distributions for each model
    if 2 in final_particles and len(final_particles[2]) > 0:
        params_model2 = np.array(final_particles[2])
        print("\nParameters for Model 2 (alpha0, n, beta, alpha):")
        print("Means:", np.mean(params_model2, axis=0))
        print("Medians:", np.median(params_model2, axis=0))
    else:
        print("\nNo particles accepted for Model 2.")

    # Plot model probabilities across populations
    plt.figure()
    for m in range(len(models)):
        probs = [len(p.get(m, []))/N for p in particles]
        plt.plot(probs, label=f'Model {m}')
    plt.xlabel('Population')
    plt.ylabel('Model Probability')
    plt.title('Model Selection Across Populations')
    plt.legend()
    plt.show() 