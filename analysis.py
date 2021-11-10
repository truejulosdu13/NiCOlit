import copy
from sklearn.base import is_classifier, is_regressor
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, balanced_accuracy_score, explained_variance_score, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def analysis_train_set_size(X, y, stratification, metric=mean_absolute_error, predictor=RandomForestRegressor(n_estimators=100),
                test_size=0.2, ticks=np.linspace(0.1, 1, 10), n_iterations_external=10, n_iterations_internal=3):
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
        #mean_prediction = np.mean(y[indexes])
        stratified_results[strat] = mean_prediction 
        
    for training_size in ticks:
        metric_current = []
        for i in range(n_iterations_external):  
            X_training, X_external_test, y_training, y_external_test, strat_training, strat_external_test = train_test_split(X, y, stratification, test_size=test_size, random_state=i)
            for j in range(n_iterations_internal):
                if training_size<1:
                    X_train, X_test, y_train, y_test, strat_train, strat_test = train_test_split(X_training, y_training, strat_training,
                    test_size=1-training_size, random_state=j)
                else:
                    X_train, y_train = X_training, y_training
                pred = copy.deepcopy(predictor)
                pred.fit(X_train, y_train)
                y_pred = pred.predict(X_external_test)
                
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

def analysis_stratification_influence(X, y, stratification, metric=mean_absolute_error, predictor=RandomForestRegressor(n_estimators=100),
                                      test_size=0.2, n_iterations=10):
    unique_stratification = np.unique(stratification)
    metric_standalone = []
    metric_augmented = []
    metric_baseline_standalone = []
    metric_baseline_augmented = []
    sizes = []
    for strat in unique_stratification:
        indexes = np.array([i for i in range(len(stratification)) if stratification[i]==strat])
        indexes_outside = np.array([i for i in range(len(stratification)) if stratification[i]!=strat])
        metric_standalone_current = []
        metric_augmented_current = []
        
        metric_baseline_standalone_current = []
        metric_baseline_augmented_current = []
        
        sizes.append(len(indexes))
        for i in range(n_iterations):
            X_training, X_external_test, y_training, y_external_test = train_test_split(X[indexes, :], y[indexes], test_size=test_size, random_state=i)
            X_outside, y_outside = X[indexes_outside, :], y[indexes_outside]
            pred = copy.deepcopy(predictor)
            pred.fit(X_training, y_training)
            y_pred = pred.predict(X_external_test)
            metric_standalone_current.append(metric(y_external_test, y_pred))
            
            if is_classifier(pred):
                values, counts = np.unique(y_training, return_counts=True)
                ind = np.argmax(counts)
                mean_prediction = np.array([values[ind] for _ in range(len(y_external_test))])
            else:
                mean_prediction = np.array([np.mean(y_training) for _ in range(len(y_external_test))])
                
            metric_baseline_standalone_current.append(metric(y_external_test, mean_prediction))
            
            pred = copy.deepcopy(predictor)
            pred.fit(np.concatenate((X_training, X_outside)), np.concatenate((y_training, y_outside)))
            y_pred = pred.predict(X_external_test)
            metric_augmented_current.append(metric(y_external_test, y_pred))
            
            if is_classifier(pred):
                values, counts = np.unique(np.concatenate((y_training, y_outside)), return_counts=True)
                ind = np.argmax(counts)
                mean_prediction = np.array([values[ind] for _ in range(len(y_external_test))])
            else:
                mean_prediction = np.array([np.mean(np.concatenate((y_training, y_outside))) for _ in range(len(y_external_test))])
            metric_baseline_augmented_current.append(metric(y_external_test, mean_prediction))
            
        metric_standalone.append(metric_standalone_current)
        metric_augmented.append(metric_augmented_current)
        metric_baseline_standalone.append(metric_baseline_standalone_current)
        metric_baseline_augmented.append(metric_baseline_augmented_current)
        
    metric_standalone = np.array(metric_standalone)
    metric_augmented = np.array(metric_augmented)
    metric_baseline_standalone = np.array(metric_baseline_standalone)
    metric_baseline_augmented = np.array(metric_baseline_augmented)
    return metric_standalone, metric_augmented, metric_baseline_standalone, metric_baseline_augmented, unique_stratification, sizes


def analysis_stratification_influence_raw(X, y, stratification, additonal_stratification, metric=mean_absolute_error, predictor=RandomForestRegressor(n_estimators=100),
                                      test_size=0.2, n_iterations=10):
    unique_stratification = np.unique(stratification)
    stratification_results = []
    additional_stratification_results = []
    local_results = []
    global_results = []
    local_baseline_results = []
    values = []
    
    for strat in unique_stratification:
        indexes = np.array([i for i in range(len(stratification)) if stratification[i]==strat])
        indexes_outside = np.array([i for i in range(len(stratification)) if stratification[i]!=strat])

        for i in range(n_iterations):
            X_training, X_external_test, y_training, y_external_test, _, additonal_stratification_external_test = train_test_split(X[indexes, :], y[indexes], additonal_stratification[indexes], test_size=test_size, random_state=i)
            X_outside, y_outside = X[indexes_outside, :], y[indexes_outside]
            pred = copy.deepcopy(predictor)
            pred.fit(X_training, y_training)
            y_pred = pred.predict(X_external_test)
            values.extend(list(y_external_test))
            if is_classifier(pred):
                values, counts = np.unique(y_training, return_counts=True)
                ind = np.argmax(counts)
                mean_prediction = [values[ind] for _ in range(len(y_external_test))]
            else:
                mean_prediction = [np.mean(y_training) for _ in range(len(y_external_test))]
                
            stratification_results.extend([strat for _ in range(len(y_external_test))])
            additional_stratification_results.extend(additonal_stratification_external_test)
            local_results.extend(list(y_pred))
            local_baseline_results.extend(mean_prediction)
            
            pred = copy.deepcopy(predictor)
            pred.fit(np.concatenate((X_training, X_outside)), np.concatenate((y_training, y_outside)))
            y_pred = pred.predict(X_external_test)
            global_results.extend(list(y_pred))

            
    return stratification_results, additional_stratification_results, local_results, global_results, local_baseline_results, values