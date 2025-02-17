import numpy as np
from scipy.integrate import odeint
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt

# Define SIR Models
def sir_basic(y, t, alpha, gamma, d, v):
    S, I, R = y
    dSdt = alpha - gamma * S * I - d * S
    dIdt = gamma * S * I - v * I - d * I
    dRdt = v * I - d * R
    return [dSdt, dIdt, dRdt]

def sir_delayed(y, t, alpha, gamma, d, v, tau):
    # Placeholder for delay implementation (requires DDE solver)
    # Using ODE as approximation for demonstration
    S, I, R = y
    dSdt = alpha - gamma * S * I - d * S
    dIdt = gamma * S * I - v * I - d * I
    dRdt = v * I - d * R
    return [dSdt, dIdt, dRdt]

def sir_latent(y, t, alpha, gamma, d, v, delta):
    S, L, I, R = y
    dSdt = alpha - gamma * S * I - d * S
    dLdt = gamma * S * I - delta * L - d * L
    dIdt = delta * L - v * I - d * I
    dRdt = v * I - d * R
    return [dSdt, dLdt, dIdt, dRdt]

def sir_reinfection(y, t, alpha, gamma, d, v, e):
    S, I, R = y
    dSdt = alpha - gamma * S * I - d * S + e * R
    dIdt = gamma * S * I - v * I - d * I
    dRdt = v * I - (d + e) * R
    return [dSdt, dIdt, dRdt]

# Simulation functions with noise
def simulate_sir_basic(theta, t_obs, noise_std=0.5):
    alpha, gamma, d, v = theta
    y0 = [20, 10, 0]  # Initial conditions: S, I, R
    sol = odeint(sir_basic, y0, t_obs, args=(alpha, gamma, d, v))
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol.flatten()

def simulate_sir_delayed(theta, t_obs, noise_std=0.5):
    alpha, gamma, d, v, tau = theta
    y0 = [20, 10, 0]
    sol = odeint(sir_delayed, y0, t_obs, args=(alpha, gamma, d, v, tau))
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol.flatten()

def simulate_sir_latent(theta, t_obs, noise_std=0.5):
    alpha, gamma, d, v, delta = theta
    y0 = [20, 0, 10, 0]  # S, L, I, R
    sol = odeint(sir_latent, y0, t_obs, args=(alpha, gamma, d, v, delta))
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol[:, [0,2,3]].flatten()  # Observe S, I, R

def simulate_sir_reinfection(theta, t_obs, noise_std=0.5):
    alpha, gamma, d, v, e = theta
    y0 = [20, 10, 0]
    sol = odeint(sir_reinfection, y0, t_obs, args=(alpha, gamma, d, v, e))
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol.flatten()

# Distance function
def distance(sim_data, observed_data):
    return np.sum((sim_data - observed_data) ** 2)

# Modified ABC SMC Algorithm for Model Selection
def abc_smc_model_selection(N, T, epsilons, models, model_prior, observed_data, t_obs):
    particles = []  # Each entry is a dict {model: array of particles}
    weights = []     # Each entry is a dict {model: array of weights}

    # Population 0: Sample from priors
    current_particles = {m: [] for m in range(len(models))}
    current_weights = {m:  [] for m in range(len(models))}
    num_particles = 0
    attempt_count = 0
    max_attempts = N * 1000
    print("\nabc 1")
    while num_particles < N and attempt_count < max_attempts:
        m = np.random.choice(len(models), p=model_prior)
        params = [prior.rvs() for prior in models[m]['priors']]
        sim_data = models[m]['simulate'](params, t_obs)
        attempt_count += 1
        if distance(sim_data, observed_data) <= epsilons[0]:
            current_particles[m].append(params)
            current_weights[m].append(1.0)
            num_particles += 1
    # Normalize weights
    if num_particles < N:
        raise RuntimeError(f"Stopped after {attempt_count} attempts. Only {num_particles} particles accepted.")
    print("\nabc 2")
    for m in current_particles:
        if current_particles[m]:
            current_weights[m] = np.array(current_weights[m]) / np.sum(current_weights[m])
    particles.append(current_particles)
    weights.append(current_weights)

    print("\nabc 3 ")

    # Subsequent populations
    for t in range(1, T):
        print(f"Processing population {t}/{T-1}")
        new_particles = {m: [] for m in range(len(models))}
        new_weights = {m: [] for m in range(len(models))}
        num_accepted = 0
        while num_accepted < N:
            m = np.random.choice(len(models), p=model_prior)
            if not particles[t-1][m]:
                continue
            idx = np.random.choice(len(particles[t-1][m]), p=weights[t-1][m])
            prev_params = particles[t-1][m][idx]
            # Perturb parameters
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

# Example usage with SIR models
if __name__ == "__main__":
    # Define SIR models
    models = [
        {   # Model 0: Basic SIR (alpha, gamma, d, v)
            'simulate': simulate_sir_basic,
            'priors': [uniform(0, 1), uniform(0, 0.1), uniform(0, 0.1), uniform(0, 0.5)],
            'kernel_sigma': [0.05, 0.01, 0.01, 0.05]
        },
        {   # Model 1: Delayed SIR (placeholder)
            'simulate': simulate_sir_delayed,
            'priors': [uniform(0, 1), uniform(0, 0.1), uniform(0, 0.1), uniform(0, 0.5), uniform(0, 2)],
            'kernel_sigma': [0.05, 0.01, 0.01, 0.05, 0.1]
        },
        {   # Model 2: Latent phase SIR
            'simulate': simulate_sir_latent,
            'priors': [uniform(0, 1), uniform(0, 0.1), uniform(0, 0.1), uniform(0, 0.5), uniform(0, 2)],
            'kernel_sigma': [0.05, 0.01, 0.01, 0.05, 0.1]
        },
        {   # Model 3: Reinfection SIR
            'simulate': simulate_sir_reinfection,
            'priors': [uniform(0, 1), uniform(0, 0.1), uniform(0, 0.1), uniform(0, 0.5), uniform(0, 0.2)],
            'kernel_sigma': [0.05, 0.01, 0.01, 0.05, 0.05]
        }
    ]
    model_prior = [0.25, 0.25, 0.25, 0.25]  # Uniform prior

    # Generate observed data from true model (Basic SIR)
    t_obs = np.linspace(0, 20, 15)
    true_params = [0.01, 0.005, 0.001, 0.1]  # alpha, gamma, d, v
    observed_data = simulate_sir_basic(true_params, t_obs, noise_std=0.5)

    # ABC SMC parameters
    N = 500  # Particles per population
    T = 4     # Populations
    epsilons = [1000, 500, 200, 100]  # Tolerance schedule
    print("\nbefore abc")

    # Run ABC SMC
    particles, weights = abc_smc_model_selection(N, T, epsilons, models, model_prior, observed_data, t_obs)

    # Analyze results
    final_particles = particles[-1]
    final_weights = weights[-1]
    model_counts = {m: len(final_particles.get(m, [])) for m in range(len(models))}
    total = sum(model_counts.values())
    print("\nModel posterior probabilities:")
    for m in range(len(models)):
        print(f"Model {m}: {model_counts[m]/total:.2%}")

    # Plot results
    plt.figure()
    for m in range(len(models)):
        probs = [len(p.get(m, []))/N for p in particles]
        plt.plot(probs, label=f'Model {m}')
    plt.xlabel('Population')
    plt.ylabel('Model Probability')
    plt.title('SIR Model Selection Across Populations')
    plt.legend()
    plt.show()