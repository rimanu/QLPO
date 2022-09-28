import numpy as np
import multiprocessing as mp
import pandas as pd
import itertools as it
import time
import random
from rlscore.learner import RLS
from rlscore.learner import LeaveOneOutRLS
from rlscore.measure import auc
__requires__= 'scikit-learn==1.0.2'
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Don't show the warnings when there are only 3 positives
import warnings
warnings.filterwarnings("ignore")

"""
Different cross-validations with RLS in library RLScore.
"""
#O(nlog(n)) time cross-validation algorithm based on quick sort algorithm
def rls_quicksort(rls, indices, generator):
    #randomly select pivot element
    pivot = generator.randint(0, len(indices)-1)
    pivot_index = indices.pop(pivot)    
    #elements smaller than pivot
    smaller = []
    #elements larger than pivot
    greater = []
    #comp_count: number of comparisons done during sorting
    comp_count = 0
    #pairwise comparison of pivot to all the other elements
    # print(type(pivot_index))
    P1, P2 = rls.leave_pair_out(indices, len(indices)*[pivot_index])
    if P1.ndim == 0:
        P1 = P1.reshape((1))
        P2 = P2.reshape((1))
    for i in range(P1.shape[0]):
        comp_count += 1
        if P1[i] < P2[i]:
            smaller.append(indices[i])
        else:
            greater.append(indices[i])
    #recursively sort part of list before the pivot
    if len(smaller) > 1:
        smaller, c = rls_quicksort(rls, smaller, generator)
        comp_count += c
    #recursively sort part of list after the pivot
    if len(greater) > 1:
        greater, c = rls_quicksort(rls, greater, generator)
        comp_count += c
    #merge together the sorted lists
    return smaller + [pivot_index] + greater, comp_count

#O(n^2) time cross-validation algorithm based on tournament
def rls_tournament(rls, n):
    #number of "wins" in the tournament for each element
    wins = 1.*np.zeros(n)
    ind1 = []
    ind2 = []
    #compare all (i,j) pairs
    for i in range(n-1):
        for j in range(i+1, n):
            ind1.append(i)
            ind2.append(j)
    P1, P2 = rls.leave_pair_out(ind1, ind2)
    for k in range(len(ind1)):
        i = ind1[k]
        j = ind2[k]
        if P1[k] > P2[k]:
            wins[i] += 1
        elif P1[k] < P2[k]:
            wins[j] += 1
        else:
            wins[i] += 0.5
            wins[j] += 0.5
    return(wins)

def rls_10fold(rls, n, splits):
    P = np.zeros(n)
    for train_index, test_index in splits:
        P[test_index] = rls.holdout(test_index)
    return P

"""
Different cross-validations with LR and RF in library scikit-learn.
"""
def get_fold_ids(n, splits):
    test_sets = [t[1] for t in splits]
    fold_ids = np.array([0]*n)
    for i in range(len(test_sets)):
        # print(test_sets[i])
        fold_ids[test_sets[i]] = i
    return(fold_ids)

def leave_pair_out(clf, splits, X, y):
    X = X.to_numpy()
    y = y.to_numpy()
    
    predictions = []
    for train_index, test_index in splits:
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)

        predictions.append(clf.predict_proba(X_test)[::,1])
        
    return pd.DataFrame(predictions)

def tournament(clf, X, y):
    n = len(y)
    # Number of wins in the tournament for each element
    wins = [0]*n
    lpo = LeavePOut(p = 2)

    # Calculate the predictions for each pair
    P = leave_pair_out(clf, lpo.split(X), X, y)
    # Calculate the numbers of wins
    for k in range(P.shape[0]):
        i = list(lpo.split(X))[k][1][0]
        j = list(lpo.split(X))[k][1][1]

        if P.loc[k][0] > P.loc[k][1]:
            wins[i] += 1
        elif P.loc[k][0] < P.loc[k][1]:
            wins[j] += 1
        else:
            wins[i] += 0.5
            wins[j] += 0.5
    return(wins)

def QS_splits(indices, generator, indices_all):
    # randomly select pivot element
    pivot = generator.randint(0, len(indices)-1)
    pivot_index = [indices.pop(pivot)]
    splits = []
    for i in it.product(indices, pivot_index):
        train_data = indices_all.copy()
        train_data.pop(train_data.index(i[0]))
        train_data.pop(train_data.index(i[1]))
        splits.append((np.array(train_data), np.array(i)))
    return(splits)

def quicksort(clf, X, y, indices, generator, indices_all):
    # Test sets consists of pairs of pivot element and the rest of indices
    splits_QS = QS_splits(indices, generator, indices_all)
    # Elements smaller than pivot
    smaller = []
    # Elements larger than pivot
    greater = []
    # Comp_count: number of comparisons done during sorting
    comp_count = 0
    # Pairwise comparison of pivot to all the other elements   
    P = leave_pair_out(clf, splits_QS, X, y)
    for k in range(P.shape[0]):
        # Which index was in test set together with the pivot element
        ind = splits_QS[k][1][0]
        comp_count += 1
        if P.loc[k][0] < P.loc[k][1]:
            smaller.append(ind)
        else:
            greater.append(ind)
    # Recursively sort part of list before the pivot
    if len(smaller) > 1:
        smaller, c = quicksort(clf, X, y, smaller, generator, indices_all)
        comp_count += c
    # Recursively sort part of list after the pivot
    if len(greater) > 1:
        greater, c = quicksort(clf, X, y, greater, generator, indices_all)
        comp_count += c
    #merge together the sorted lists
    return smaller + [splits_QS[0][1][1]] + greater, comp_count

"""
The simulation study.
"""
def simulation_calculations(data, seed_split):
    seed = seed_split[0].generate_state(1)[0]
    split = seed_split[1]
    train_indices = data.index.isin(split[0])
    test_indices = data.index.isin(split[1])
    
    X = data.iloc[train_indices, 3:]
    y = data.Class.loc[train_indices]
    X_test = data.iloc[test_indices, 3:]
    y_test = data.Class.loc[test_indices]
    
    generator = random.Random(seed)
    indices_all = list(range(X.shape[0]))

    # Models
    rls = RLS(X, y.astype("double"), regparam = 1, bias = 1)
    
    rf_test =  RandomForestClassifier(random_state=seed, n_estimators = 100)
    rf =  RandomForestClassifier(random_state=seed, n_estimators = 100)
    
    lr_test = LogisticRegression(penalty = "l2", random_state=seed, solver = "liblinear", max_iter = 100)
    lr = LogisticRegression(penalty = "l2", random_state=seed, solver = "liblinear", max_iter = 100)
    
    # Independent test set predictions
    P_RLS_test = rls.predict(X_test)
    
    P_RF_test = rf_test.fit(X, y).predict_proba(X_test)[::,1]
    
    P_LR_test = lr_test.fit(X, y).predict_proba(X_test)[::,1]
    
    test_label_pred = pd.DataFrame(data = [[seed] + ["y_test"] + list(y_test), [seed] + ["RLS_test"] + list(P_RLS_test), [seed] + ["RF_test"] + list(P_RF_test), [seed] + ["LR_test"] + list(P_LR_test)])
    
    # Quicksort
    I_RLS, comp_count_RLS = rls_quicksort(rls, indices_all.copy(), generator)
    P_RLS_QLPO = np.argsort(I_RLS)
    
    I_RF, comp_count_RF = quicksort(rf, X, y, indices_all.copy(), generator, indices_all)
    P_RF_QLPO = np.argsort(I_RF)
    
    I_LR, comp_count_LR = quicksort(lr, X, y, indices_all.copy(), generator, indices_all)
    P_LR_QLPO = np.argsort(I_LR)
    
    # TLPO
    P_RLS_TLPO = rls_tournament(rls, X.shape[0])
    
    P_RF_TLPO = tournament(rf, X, y)
    
    P_LR_TLPO = tournament(lr, X, y)
    
    # 10-fold
    kfold = StratifiedKFold(n_splits = 10, random_state = seed, shuffle = True)
    kfold_splits = list(kfold.split(X,y))
    fold_ids = get_fold_ids(len(indices_all), kfold_splits)
    P_RLS_10fold = rls_10fold(rls, len(indices_all), kfold_splits)
    
    P_RF_10fold = cross_val_predict(rf, X, y, cv = kfold, method="predict_proba")[::,1].tolist()
    
    P_LR_10fold = cross_val_predict(lr, X, y, cv = kfold, method="predict_proba")[::,1].tolist()
    
    # LOO
    learner_RLS = LeaveOneOutRLS(X, y, regparams=[1], measure=auc)
    P_RLS_LOO = learner_RLS.cv_predictions
    
    loo = LeaveOneOut()
    P_RF_LOO = cross_val_predict(rf, X, y, cv = loo, method="predict_proba")[::,1].tolist()
    
    P_LR_LOO = cross_val_predict(lr, X, y, cv = loo, method="predict_proba")[::,1].tolist()

    results = pd.DataFrame(data = [[seed] + ["index"] + X.index.to_list(), [seed]+["y"]+list(y), [seed] + ["P10F_folds"] + list(fold_ids), [seed] + ["RLS_QLPO"] + list(P_RLS_QLPO),[seed] + ["RLS_TLPO"] + list(P_RLS_TLPO), [seed] + ["RLS_P10F"] + list(P_RLS_10fold), [seed] + ["RLS_LOO"] + list(P_RLS_LOO[0]), [seed] + ["RF_QLPO"] + list(P_RF_QLPO),[seed] + ["RF_TLPO"] + list(P_RF_TLPO), [seed] + ["RF_P10F"] + P_RF_10fold, [seed] + ["RF_LOO"] + P_RF_LOO, [seed] + ["LR_QLPO"] + list(P_LR_QLPO),[seed] + ["LR_TLPO"] + list(P_LR_TLPO), [seed] + ["LR_P10F"] + P_LR_10fold, [seed] + ["LR_LOO"] + P_LR_LOO])
    return results, test_label_pred


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
    repetitions = 10**3
    population_size = 10**6
    test_size = 10**4
    pos_fract = [0.1, 0.5]
    n_sample = [30, 100]
    
    for theta in theta_list:
        # Import the population data.
        population_state = base_sequence.generate_state(1)[0]
        population_data_df = pd.read_csv('./population_data_mp'+str(mixing_probability)+'_theta'+str(theta)+'_popsizeE'+str(np.log10(population_size))+'.csv')
        
        # Sets of indices: all, positives and negatives
        all_indices = set(population_data_df.index)
        # Take an independent test sample
        rng = random.Random(population_state)
        test_set_indices = set(rng.sample(all_indices, k = test_size))
        
        # Indices available for taking a small sample for cross-validation
        cv_available = all_indices.difference(test_set_indices)
        cv_available_data = population_data_df[population_data_df.index.isin(cv_available)]
        positives = set(cv_available_data.index[cv_available_data.Class == 1])
        negatives = set(cv_available_data.index[cv_available_data.Class == -1])
        
        for ns in n_sample:
            for pf in pos_fract:
                n_positives = int(pf*ns)
                n_negatives = ns-n_positives
                used_indices = set()
                splits = []

                # Create the data splits
                for i in range(repetitions):
                    i_positives = set(rng.sample(positives.difference(used_indices), k = n_positives))
                    i_negatives = set(rng.sample(negatives.difference(used_indices), k = n_negatives))
                    
                    cv_indices = i_positives.union(i_negatives)
                    splits.append(tuple([cv_indices, test_set_indices]))
                    used_indices = used_indices.union(cv_indices)
                    
                # Generators to pass further
                child_seeds = np.random.SeedSequence(population_state).spawn(repetitions)
                parameters = it.product([population_data_df], list(zip(child_seeds, splits)))
                
                # Compute in parallel
                pool = mp.Pool(processes = mp.cpu_count())
                output = pool.starmap(simulation_calculations, list(parameters))
                pool.close()
                pool.join()
                
                results_list = []
                test_results_list = []
                for o in output:
                    results_list.append(o[0])
                    test_results_list.append(o[1])
                
                results = pd.concat(results_list, ignore_index = True)
                test_results = pd.concat(test_results_list, ignore_index = True)
                
                results.to_csv('./results_mp'+str(mixing_probability)+'_ns'+str(ns)+'_pf'+str(pf)+'_theta'+str(theta)+'.csv', index = False)
                test_results.to_csv('./test_results_mp'+str(mixing_probability)+'_ns'+str(ns)+'_pf'+str(pf)+'_theta'+str(theta)+'.csv', index = False)