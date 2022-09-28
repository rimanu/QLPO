import numpy as np
import pandas as pd

#### Functions for the nonlinear generative model ####
def expit(X_row, b, theta = 0.5):
    weighted_lin_and_non_lin_parts = theta*np.dot(X_row, b) + (1-theta)*(X_row[0]**(2) + X_row[1]**(2) + 4*X_row[0]*X_row[1])
    pi = np.exp(weighted_lin_and_non_lin_parts) / (1+ np.exp(weighted_lin_and_non_lin_parts))
    return pi

def generative_model(sample_size, n_features, mu, sigma, mixing_probability, beta, theta, rand_seed):
    rand_state = np.random.RandomState(rand_seed)
    Z_signs = rand_state.choice([1,-1], size = sample_size, replace = True, p = [mixing_probability, 1-mixing_probability])
    n_Z_pos = np.sum(Z_signs == 1)
    n_Z_neg = np.sum(Z_signs == -1)
    Z = np.ones(n_features)

    X_Z_pos = rand_state.multivariate_normal(mu*Z, sigma*np.eye(n_features), n_Z_pos)

    X_Z_neg = rand_state.multivariate_normal(-1*mu*Z, sigma*np.eye(n_features), n_Z_neg)

    X = np.vstack((X_Z_pos, X_Z_neg))
    pi = np.apply_along_axis(expit, 1, X, beta, theta)
    
    y = []
    for prob in pi:
        y.append(rand_state.choice([1,-1], size = 1, replace = True, p = [prob, 1-prob]))
    y = np.array(y).flatten()
    
    return X,y

if __name__ == "__main__":
    # Define the parameters
    mixing_probability = 0.25
    mu = 0.5
    sigma = 1
    n_features = 10
    theta_list = np.array(list(range(0, 5)))/4
    beta = np.array([2,1,1,1,1,0,0,0,0,0])
    random_seed = 12345
    base_sequence = np.random.SeedSequence(random_seed)
    population_size = 10**6
    for theta in theta_list:
        population_state = base_sequence.generate_state(1)[0]
        X_population, y_population = generative_model(population_size, n_features, mu, sigma, mixing_probability, beta, theta, population_state)
        X_population_df = pd.DataFrame(X_population, columns = ['Feature1','Feature2','Feature3','Feature4','Feature5','Feature6','Feature7','Feature8','Feature9',"Feature10"])
        rs_y_population_df = pd.DataFrame({"populationID" : population_state, "Class" : y_population})
        population_data_df = rs_y_population_df.join(X_population_df)
        # Save the generated population data
        population_data_df.to_csv('./population_data_mp'+str(mixing_probability)+'_theta'+str(theta)+'_popsizeE'+str(np.log10(population_size))+'.csv', index = False)