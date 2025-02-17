import numpy as np
from scipy.integrate import odeint
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt


# Lotka-Volterra model
def lotka_volterra(y, t, a, b):
    x, y_pred = y
    dxdt = a * x - x * y_pred
    dydt = b * x * y_pred - y_pred
    return [dxdt, dydt]

# Simulate data using the model
def simulate_data(theta, t_obs, noise_std=0.5):
    a, b = theta
    y0 = [1.0, 0.5]  # Initial conditions
    sol = odeint(lotka_volterra, y0, t_obs, args=(a, b))
    # Add Gaussian noise
    noisy_sol = sol + np.random.normal(0, noise_std, sol.shape)
    return noisy_sol.flatten()  # Flatten for distance calculation

def distance(sim_data, observed_data):
    return np.sum((sim_data - observed_data) ** 2)


# ABC SMC Algorithm
def abc_smc(N, T, epsilons, prior_a, prior_b, kernel_sigma, observed_data, t_obs):
    # Initialize
    particles = []
    weights = []

    # Sample from prior
    current_particles = []
    current_weights = []
    while len(current_particles) < N:
        a = prior_a.rvs()
        b = prior_b.rvs()
        sim_data = simulate_data((a, b), t_obs)
        if distance(sim_data, observed_data) <= epsilons[0]:
            current_particles.append([a, b])
            current_weights.append(1.0)
    # Normalize weights
    current_weights = np.array(current_weights) / np.sum(current_weights)
    particles.append(np.array(current_particles))
    weights.append(current_weights)

    #S3
    for t in range(1, T):
        print(f"Processing population {t}/{T - 1}")
        new_particles = []
        new_weights = []
        for _ in range(N):
            # Sample from previous population
            idx = np.random.choice(len(particles[t - 1]), p=weights[t - 1])
            a_prev, b_prev = particles[t - 1][idx]

            # Perturb
            a_proposed = a_prev + np.random.normal(0, kernel_sigma[0])
            b_proposed = b_prev + np.random.normal(0, kernel_sigma[1])

            # Check prior
            if not (prior_a.ppf(0) <= a_proposed <= prior_a.ppf(1) and
                    prior_b.ppf(0) <= b_proposed <= prior_b.ppf(1)):
                continue

            sim_data = simulate_data((a_proposed, b_proposed), t_obs)
            if distance(sim_data, observed_data) <= epsilons[t]:
                # Calculate weight
                prior_density = prior_a.pdf(a_proposed) * prior_b.pdf(b_proposed)
                kernel_density = (
                        norm(a_prev, kernel_sigma[0]).pdf(a_proposed) *
                        norm(b_prev, kernel_sigma[1]).pdf(b_proposed)
                )
                denominator = np.sum([
                    weights[t - 1][i] *
                    norm(particles[t - 1][i][0], kernel_sigma[0]).pdf(a_proposed) *
                    norm(particles[t - 1][i][1], kernel_sigma[1]).pdf(b_proposed)
                    for i in range(len(particles[t - 1]))
                ])
                weight = prior_density / (denominator + 1e-12)  # Avoid division by zero
                new_particles.append([a_proposed, b_proposed])
                new_weights.append(weight)
        # Normalize weights
        if new_particles:
            new_weights = np.array(new_weights) / np.sum(new_weights)
            particles.append(np.array(new_particles))
            weights.append(new_weights)
        else:
            # If no particles accepted
            particles.append(particles[t - 1])
            weights.append(weights[t - 1])
    return particles, weights


# Example usage
if __name__ == "__main__":
    # Generate observed data (true parameters: a=1.0, b=1.0)
    t_obs = np.linspace(0, 15, 8)
    true_theta = (1.0, 1.0)
    observed_data = simulate_data(true_theta, t_obs, noise_std=0.0)  # No noise for testing

    # ABC SMC parameters
    N = 1000 # Number of particles per population
    T = 5  # Number of populations
    epsilons = [30.0, 16.0, 6.0, 5.0, 4.3]
    prior_a = uniform(loc=0.5, scale=1.5)  # Uniform prior for a
    prior_b = uniform(loc=0.5, scale=1.5)  # Uniform prior for b
    kernel_sigma = [0.1, 0.1]  # Perturbation kernel

    # Run ABC SMC
    particles, weights = abc_smc(N, T, epsilons, prior_a, prior_b, kernel_sigma, observed_data, t_obs)

    # Results from the last population
    final_particles = particles[-1]
    final_weights = weights[-1]
    print("Final particles (a, b):")
    print(final_particles)
    print("Median estimates:", np.median(final_particles, axis=0))


    def plot_results(observed_data, true_trajectory, t_obs, particles, weights, populations_to_plot=[0, -1]):

        plt.figure(figsize=(12, 6))

        # Plot observed data (noisy points)
        prey_obs = observed_data[::2]  # Even indices: prey
        predator_obs = observed_data[1::2]  # Odd indices: predator
        plt.scatter(t_obs, prey_obs, c='blue', marker='o', label='Observed Prey', alpha=0.7)
        plt.scatter(t_obs, predator_obs, c='red', marker='^', label='Observed Predator', alpha=0.7)

        # Plot true trajectory (without noise)
        prey_true = true_trajectory[::2]
        predator_true = true_trajectory[1::2]
        plt.plot(t_obs, prey_true, 'b-', lw=2, label='True Prey')
        plt.plot(t_obs, predator_true, 'r-', lw=2, label='True Predator')

        # Plot simulations from selected populations
        colors = ['gray', 'green']
        for i, pop_idx in enumerate(populations_to_plot):
            population = particles[pop_idx]
            weights_pop = weights[pop_idx]

            # Plot 10 random particles from the population
            for _ in range(10):
                idx = np.random.choice(len(population), p=weights_pop)
                a, b = population[idx]
                sim = simulate_data((a, b), t_obs, noise_std=0.0)  # Simulate without noise
                prey_sim = sim[::2]
                predator_sim = sim[1::2]
                linestyle = '--' if pop_idx == 0 else ':'
                alpha = 0.3 if pop_idx == 0 else 0.5
                label = f'Pop {pop_idx} Samples' if _ == 0 else None
                plt.plot(t_obs, prey_sim, color=colors[i], linestyle=linestyle, alpha=alpha, label=label)
                plt.plot(t_obs, predator_sim, color=colors[i], linestyle=linestyle, alpha=alpha)

        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title('ABC SMC: Observed vs Simulated Trajectories')
        plt.legend()
        plt.show()


    # Modified main section to include true trajectory and plotting
    if __name__ == "__main__":
        # Generate REAL and observed data
        t_obs = np.linspace(0, 15, 8)
        true_theta = (1.0, 1.0)
        true_trajectory = simulate_data(true_theta, t_obs, noise_std=0.0)
        observed_data = simulate_data(true_theta, t_obs, noise_std=0.5)  # Add noise

        # ABC SMC parameters
        N = 100
        T = 5
        epsilons = [30.0, 16.0, 6.0, 5.0, 4.3]
        prior_a = uniform(loc=0.5, scale=1.5)
        prior_b = uniform(loc=0.5, scale=1.5)
        kernel_sigma = [0.1, 0.1]

        # Run ABC SMC
        particles, weights = abc_smc(N, T, epsilons, prior_a, prior_b, kernel_sigma, observed_data, t_obs)

        # Plot results
        plot_results(observed_data, true_trajectory, t_obs, particles, weights, populations_to_plot=[0, -1])
        plt.figure(figsize=(12, 6))
        for t in range(len(particles)):
            if len(particles[t]) > 0:
                plt.scatter(particles[t][:, 0], particles[t][:, 1],
                            alpha=0.5, label=f'Population {t}', s=10)
        plt.xlabel('Parameter a')
        plt.ylabel('Parameter b')
        plt.title('ABC SMC: Parameter Distributions Across Populations')
        plt.axvline(x=1.0, color='k', linestyle='--')
        plt.axhline(y=1.0, color='k', linestyle='--')
        plt.legend()
        plt.show()