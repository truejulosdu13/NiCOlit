import copy
from sklearn.base import is_classifier, is_regressor
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, balanced_accuracy_score, explained_variance_score, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def analysis_train_set_size(X, y, stratification, metric=mean_absolute_error, predictor=RandomForestRegressor(n_estimators=100),
                test_size=0.2, ticks=np.linspace(0.1, 1, 10), n_iterations_external=10, n_iterations_internal=3):
    
    """Analyze the evolution of a given performance metric when the size of the training set varies. This analysis is performed 
    at fixed test set. The experiment is ran over multiple test sets, and the results aggregated. 
            Parameters:
                    X (np array): features of the dataset, of shape (n_samples, n_features) 
                    y (np array): labels of the dataset
                    stratification (np.array): additional labels to use for the baseline
                    metric (sklearn.metrics): performance metric
                    predictor (sklearn regression model): predictive model
                    test_size (float): test size (between 0 and 1) at which to perform the analysis 
                    ticks (np array): arrays of train sizes (between 0 and 1) at which to perform the analysis
                    n_iterations_external (int): number of iterations on different test sets 
                    n_iterations_internal (int): number of iterations at fixed test size

            Returns:
                    metric_values (np array): results obtained with the model
                    baseline_values (np array): results obtained with the baseline
                    sizes (np array): corresponding training set size 
    """
    
    metric_values = []
    baseline_values = []
    metric_current = [] 
    baseline_current = []
    sizes = []
    
    # Getting the dummy baseline 
    stratified_results = {}
    for strat in np.unique(stratification):
        indexes = np.array([i for i in range(len(stratification)) if stratification[i]==strat])
        if is_classifier(predictor):
            values, counts = np.unique(y[indexes], return_counts=True)
            ind = np.argmax(counts)
            mean_prediction = np.array(values[ind])
        else:
            mean_prediction = np.mean(y[indexes])
            
        stratified_results[strat] = mean_prediction 
    
    # Iterate over training sizes
    for training_size in ticks:
        # Track metric values for current training size 
        metric_current = []
        # Iterate over test sets 
        for i in range(n_iterations_external):  
            X_training, X_external_test, y_training, y_external_test, strat_training, strat_external_test = train_test_split(X, y, stratification, test_size=test_size, random_state=i)
            # Iterate over internal training sets 
            for j in range(n_iterations_internal):
                if training_size<1:
                    X_train, X_test, y_train, y_test, strat_train, strat_test = train_test_split(X_training, y_training, strat_training,
                    test_size=1-training_size, random_state=j)
                else:
                    X_train, y_train = X_training, y_training
                # Train the model and make predictions 
                pred = copy.deepcopy(predictor)
                pred.fit(X_train, y_train)
                y_pred = pred.predict(X_external_test)
                
                # Get baseline predictions 
                dummy_predictions = []
                for s in strat_external_test:
                    dummy_predictions.append(stratified_results[s])
                metric_current.append(metric(y_external_test, y_pred))
                baseline_current.append(metric(y_external_test, dummy_predictions))
        sizes.append(len(y_train))
        metric_values.append(metric_current)
        baseline_values.append(baseline_current)
    metric_values = np.array(metric_values)
    baseline_values = np.array(baseline_values)
    return metric_values, baseline_values, sizes

def random_split(X, y, stratification, additional_stratification, predictor=RandomForestRegressor(n_estimators=100),
                test_size=0.2, n_iterations=1):
    
    """Gathers the prediction of a regression model on various random splits. Includes a baseline based on a given stratification, and 
     keeps track of an additional stratification parameter (e.g. scope/optimisation origin of the reaction).
            Parameters:
                    X (np array): features of the dataset, of shape (n_samples, n_features) 
                    y (np array): labels of the dataset
                    stratification (np.array): additional labels to use for the baseline
                    additional_stratification (np.array): additional labels that we need to keep track of 
                    predictor (sklearn regression model): predictive model
                    test_size (float): test size (between 0 and 1) at which to perform the analysis 
                    n_iterations (int): number of iterations
            Returns:
                    values (np array): actual yields 
                    baseline_values (np array): results obtained with the baseline
                    model_values (np array): results obtained with the model
                    stratification_values (np array): stratification_values
                    additional_stratification_values (np array): additional_stratification_values
    """
    
    values = []
    baseline_values = []
    model_values = [] 
    stratification_values = []
    additional_stratification_values = []
    
    # Getting the dummy baseline 
    stratified_results = {}
    for strat in np.unique(stratification):
        indexes = np.array([i for i in range(len(stratification)) if stratification[i]==strat])
        if is_classifier(predictor):
            values, counts = np.unique(y[indexes], return_counts=True)
            ind = np.argmax(counts)
            mean_prediction = np.array(values[ind])
        else:
            mean_prediction = np.mean(y[indexes])
        stratified_results[strat] = mean_prediction 
    
    # Iterate over test sets  
    for i in range(n_iterations):
        X_training, X_external_test, y_training, y_external_test, strat_training, strat_external_test, _, additional_strat_external_test = train_test_split(X, y, stratification, additional_stratification, test_size=test_size, random_state=i)
        
        # Train the model and get predictions 
        pred = copy.deepcopy(predictor)
        pred.fit(X_training, y_training)
        y_pred = pred.predict(X_external_test)
        
        # Get baseline predictions
        dummy_predictions = []
        for s in strat_external_test:
            dummy_predictions.append(stratified_results[s])
        
        values.extend(y_external_test)
        baseline_values.extend(dummy_predictions)
        model_values.extend(y_pred)
        stratification_values.extend(strat_external_test)
        additional_stratification_values.extend(additional_strat_external_test)
        
        
    return values, baseline_values, model_values, stratification_values, additional_stratification_values

def stratified_split(X, y, stratification, additonal_stratification, metric=mean_absolute_error, predictor=RandomForestRegressor(n_estimators=100),
                                      test_size=0.2, n_iterations=10):
    
    """Gathers the prediction of a regression model on stratified splits. The baseline, for a given strata, is defined as the mean value of
    y for this strata. Also keeps track of an additional stratification parameter (e.g. scope/optimisation origin of the reaction).
            Parameters:
                    X (np array): features of the dataset, of shape (n_samples, n_features) 
                    y (np array): labels of the dataset
                    stratification (np.array): additional labels to use for the splits
                    additional_stratification (np.array): additional labels that we need to keep track of 
                    predictor (sklearn regression model): predictive model
                    test_size (float): test size (between 0 and 1) at which to perform the analysis 
                    n_iterations (int): number of iterations
            Returns:
                    values (np array): actual yields 
                    baseline_values (np array): results obtained with the baseline
                    model_values (np array): results obtained with the model
                    stratification_values (np array): stratification_values
                    additional_stratification_values (np array): additional_stratification_values
    """
    unique_stratification = np.unique(stratification)
    stratification_values = []
    additional_stratification_values = []
    model_values = []
    baseline_values = []
    values = []
    
    # Getting the dummy baseline 
    for strat in unique_stratification:
        indexes = np.array([i for i in range(len(stratification)) if stratification[i]==strat])
        indexes_outside = np.array([i for i in range(len(stratification)) if stratification[i]!=strat])
        
        # Iterate over test sets  
        for i in range(n_iterations):
            X_external_test, y_external_test, additonal_stratification_external_test = X[indexes, :], y[indexes], additonal_stratification[indexes]
            X_outside, y_outside = X[indexes_outside, :], y[indexes_outside]
            
            # Train the model and get predictions 
            pred = copy.deepcopy(predictor)
            pred.fit(X_outside, y_outside)
            y_pred = pred.predict(X_external_test)
            model_values.extend(list(y_pred))
            
            # Get baseline predictions
            if is_classifier(pred):
                values, counts = np.unique(y_outside, return_counts=True)
                ind = np.argmax(counts)
                mean_prediction = [values[ind] for _ in range(len(y_external_test))]
            else:
                mean_prediction = [np.mean(y_outside) for _ in range(len(y_external_test))]
                
            stratification_values.extend([strat for _ in range(len(y_external_test))])
            additional_stratification_values.extend(additonal_stratification_external_test)
            baseline_values.extend(mean_prediction)
            values.extend(list(y_external_test))
           
            

    return values, baseline_values, model_values, stratification_values, additional_stratification_values