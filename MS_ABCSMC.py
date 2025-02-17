import numpy as np
from scipy.integrate import odeint
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt

# Define two different Lotka-Volterra models for demonstration
def lotka_volterra_model1(y, t, a, b):
    """Standard LV model with parameters a, b."""
    x, y_pred = y
    dxdt = a * x - x * y_pred
    dydt = b * x * y_pred - y_pred
    return [dxdt, dydt]

def lotka_volterra_model2(y, t, a, b, c):
    """Modified LV model with an additional parameter c."""
    x, y_pred = y
    dxdt = a * x - c * x * y_pred
    dydt = b * x * y_pred - y_pred
    return [dxdt, dydt]

def simulate_model1(theta, t_obs, noise_std=0.5):
    a, b = theta
    y0 = [1.0, 0.5]
    sol = odeint(lotka_volterra_model1, y0, t_obs, args=(a, b))
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol.flatten()

def simulate_model2(theta, t_obs, noise_std=0.5):
    a, b, c = theta
    y0 = [1.0, 0.5]
    sol = odeint(lotka_volterra_model2, y0, t_obs, args=(a, b, c))
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol.flatten()

# Distance function (sum of squared errors)
def distance(sim_data, observed_data):
    return np.sum((sim_data - observed_data) ** 2)

# ABC SMC Algorithm for Model Selection
def abc_smc_model_selection(N, T, epsilons, models, model_prior, observed_data, t_obs):
    particles = []  # Each entry is a dict {model: array of particles}
    weights = []     # Each entry is a dict {model: array of weights}

    # Population 0: Sample from priors
    current_particles = {m: [] for m in range(len(models))}
    current_weights = {m: [] for m in range(len(models))}
    num_particles = 0
    while num_particles < N:
        m = np.random.choice(len(models), p=model_prior)
        params = [prior.rvs() for prior in models[m]['priors']]
        sim_data = models[m]['simulate'](params, t_obs)
        if distance(sim_data, observed_data) <= epsilons[0]:
            current_particles[m].append(params)
            current_weights[m].append(1.0)
            num_particles += 1
    # Normalize weights
    for m in current_particles:
        if current_particles[m]:
            current_weights[m] = np.array(current_weights[m]) / np.sum(current_weights[m])
    particles.append(current_particles)
    weights.append(current_weights)

    # Subsequent populations
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
            if distance(sim_data, observed_data) <= epsilons[t]:
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
    # Define models (here: Model 0 is true model)
    models = [
        {   # Model 0: Standard LV with a, b
            'simulate': simulate_model1,
            'priors': [uniform(loc=0.5, scale=1.5), uniform(loc=0.5, scale=1.5)],
            'kernel_sigma': [0.1, 0.1]
        },
        {   # Model 1: Modified LV with a, b, c (extra parameter)
            'simulate': simulate_model2,
            'priors': [uniform(loc=0.5, scale=1.5), uniform(loc=0.5, scale=1.5), uniform(loc=0.1, scale=0.3)],
            'kernel_sigma': [0.1, 0.1, 0.05]
        }
    ]
    model_prior = [0.5, 0.5]  # Uniform prior over models

    # Generate observed data (from Model 0 with a=1.0, b=1.0)
    t_obs = np.linspace(0, 15, 8)
    true_theta_model0 = (1.0, 1.0)
    observed_data = simulate_model1(true_theta_model0, t_obs, noise_std=0.5)

    # ABC SMC parameters
    N = 500  # Number of particles per population
    T = 5     # Number of populations
    epsilons = [30.0, 16.0, 6.0, 5.0, 4.3]

    # Run ABC SMC for model selection
    particles, weights = abc_smc_model_selection(N, T, epsilons, models, model_prior, observed_data, t_obs)

    # Analyze results
    final_particles = particles[-1]
    final_weights = weights[-1]
    model_counts = {m: len(final_particles.get(m, [])) for m in range(len(models))}
    total = sum(model_counts.values())
    print("\nModel posterior probabilities:")
    for m in range(len(models)):
        print(f"Model {m}: {model_counts[m]/total:.2%}")

    # Plotting parameter distributions for each model (example for Model 0)
    if 0 in final_particles:
        params_model0 = np.array(final_particles[0])
        print("\nParameters for Model 0 (a, b):")
        print("Means:", np.mean(params_model0, axis=0))
        print("Medians:", np.median(params_model0, axis=0))

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