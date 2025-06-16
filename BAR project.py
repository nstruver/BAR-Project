###Author: Nathan Struver
#Purpose: Analyze Bayesian Adaptive Randomization and compare it to other methods
#AI and Other Tools: ChatGPT was used for help with ML libraries and adding docstrings(reviewed for correctness).
#    the rest of the code came from either geeksforgeeks.com, stackoverflow, or youtube.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import bernoulli
import shap
import pandas as pd

np.random.seed(42)

n_patients = 500
n_arms = 4
true_success_rates = [0.3, 0.5, 0.4, 0.35]
prior_params = [(1, 1), (3, 2), (2, 2), (1, 3)]
strategies = ['uniform', 'bayesian', 'ml_guided']
feature_names = ['Covariate 1', 'Covariate 2', 'Covariate 3']


def generate_patient_features(n):
    """
  Generate synthetic patient covariate data.

  Parameters:
  n (int): Number of patients.

  Returns:
  np.ndarray: An (n x 3) array of patient features drawn from a standard normal distribution.
  """
    return np.random.normal(size=(n, 3))

def simulate_outcome(treatment, patient_features):
    """
    Simulate the binary outcome of a treatment for a patient.
    
    Parameters:
    treatment (int): The treatment arm chosen.
    patient_features (np.ndarray): The feature vector of the patient.
    
    Returns:
    int: 1 for success, 0 for failure, sampled from a Bernoulli distribution.
    """
    base_prob = true_success_rates[treatment]
    modifier = 0.05 * patient_features[0]
    final_prob = np.clip(base_prob + modifier, 0.05, 0.95)
    return bernoulli.rvs(final_prob)

# Store ML-guided training data separately for SHAP analysis later
ml_data = {
    'X_train': [],
    'y_train': [],
    't_train': []
}

results = {}

for strategy in strategies:
    print(f"Simulating strategy: {strategy}")
    allocations = []
    outcomes = []
    features = []
    
    successes = np.zeros(n_arms)
    failures = np.zeros(n_arms)
    
    X_train = []
    y_train = []
    t_train = []
    
    for i in range(n_patients):
        patient_features = generate_patient_features(1)[0]
        
        if strategy == 'uniform':
            chosen_arm = np.random.choice(n_arms)
        
        elif strategy == 'bayesian':
            sampled_probs = [np.random.beta(prior_params[j][0] + successes[j],
                                            prior_params[j][1] + failures[j])
                             for j in range(n_arms)]
            chosen_arm = np.argmax(sampled_probs)
        
        elif strategy == 'ml_guided':
            if len(X_train) < 20:
                chosen_arm = np.random.choice(n_arms)
            else:
                X = np.array(X_train)
                y = np.array(y_train)
                t = np.array(t_train)
                
                preds = []
                for arm in range(n_arms):
                    mask = t == arm
                    if np.sum(mask) > 5:
                        model = GradientBoostingClassifier().fit(X[mask], y[mask])
                        pred_prob = model.predict_proba(patient_features.reshape(1, -1))[0][1]
                    else:
                        pred_prob = 0.25
                    preds.append(pred_prob)
                chosen_arm = np.argmax(preds)
        
        outcome = simulate_outcome(chosen_arm, patient_features)
        
        if outcome == 1:
            successes[chosen_arm] += 1
        else:
            failures[chosen_arm] += 1
        
        allocations.append(chosen_arm)
        outcomes.append(outcome)
        features.append(patient_features)
        
        X_train.append(patient_features)
        y_train.append(outcome)
        t_train.append(chosen_arm)

        # Save ML training data only for ML-guided strategy
        if strategy == 'ml_guided':
            ml_data['X_train'].append(patient_features)
            ml_data['y_train'].append(outcome)
            ml_data['t_train'].append(chosen_arm)
    
    results[strategy] = {
        'allocations': allocations,
        'outcomes': outcomes,
        'successes': successes,
        'failures': failures
    }

# Plot allocation results
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for i, strategy in enumerate(strategies):
    ax[i].hist(results[strategy]['allocations'], bins=np.arange(n_arms+1)-0.5, rwidth=0.7)
    ax[i].set_title(f"{strategy.capitalize()} allocation")
    ax[i].set_xticks(range(n_arms))
    ax[i].set_xticklabels(['A', 'B', 'C', 'D'])
    ax[i].set_xlabel("Arm")
    ax[i].set_ylabel("Patients Assigned")
    ax[i].grid(True)
plt.tight_layout()
plt.show()

# Plot success and regret
best_rate = max(true_success_rates)
plt.figure(figsize=(12,5))

for strategy in strategies:
    allocations = np.array(results[strategy]['allocations'])
    outcomes = np.array(results[strategy]['outcomes'])
    
    cum_success = np.cumsum(outcomes)
    chosen_rates = np.array([true_success_rates[a] for a in allocations])
    regret = best_rate - chosen_rates
    cum_regret = np.cumsum(regret)
    
    prop_best_arm = np.mean(allocations == np.argmax(true_success_rates))
    
    print(f"\n=== Policy Evaluation for {strategy.upper()} ===")
    print(f"Total Successes: {cum_success[-1]} out of {n_patients} patients")
    print(f"Overall Success Rate: {100 * cum_success[-1] / n_patients:.2f}%")
    print(f"Total Cumulative Regret: {cum_regret[-1]:.2f}")
    print(f"Proportion of Patients Assigned to Best Arm: {100 * prop_best_arm:.2f}%")
    
    plt.subplot(1, 2, 1)
    plt.plot(cum_success, label=strategy.capitalize())
    plt.title('Cumulative Successes Over Patients')
    plt.xlabel('Patient Number')
    plt.ylabel('Cumulative Successes')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(cum_regret, label=strategy.capitalize())
    plt.title('Cumulative Regret Over Patients')
    plt.xlabel('Patient Number')
    plt.ylabel('Cumulative Regret')
    plt.grid(True)

plt.subplot(1, 2, 1)
plt.legend(title='Strategy')
plt.subplot(1, 2, 2)
plt.legend(title='Strategy')
plt.tight_layout()
plt.show()

#SHAP analysis
X_train = np.array(ml_data['X_train'])
y_train = np.array(ml_data['y_train'])
t_train = np.array(ml_data['t_train'])

arm_labels = ['A', 'B', 'C', 'D']

for arm in range(n_arms):
    mask = t_train == arm
    if np.sum(mask) > 5:
        print(f"\nTraining SHAP model for Arm {arm} ({np.sum(mask)} samples)...")
        
        model = GradientBoostingClassifier().fit(X_train[mask], y_train[mask])
        explainer = shap.Explainer(model, X_train[mask], feature_names=feature_names)
        shap_values = explainer(X_train[mask])
        
        plt.figure()
        plt.title(f"SHAP Summary Plot for Arm {arm_labels[arm]} (Treatment {arm + 1})")
        shap.summary_plot(shap_values, X_train[mask], feature_names=feature_names, show=True)
        
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        df_shap = pd.DataFrame({
            'Feature': feature_names,
            'Mean |SHAP|': mean_abs_shap
        }).sort_values(by='Mean |SHAP|', ascending=False)
        
        print(f"Average absolute SHAP values for Arm {arm_labels[arm]}:")
        print(df_shap.to_string(index=False))
    else:
        print(f"\nNot enough data for Arm {arm_labels[arm]} to train SHAP model (only {np.sum(mask)} samples).")
        
