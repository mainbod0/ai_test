import optuna

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

iris = load_iris()
x, y = iris.data, iris.target

# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def my_objective(trial):

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = SVC(C=svc_c, gamma="auto")
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )

    score = cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3, scoring='accuracy')
    accuracy = score.mean()
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(my_objective, n_trials=100)
    print(study.best_trial)
    study.best_value
    # 0.9866666666666667
    study.best_params
    # {'classifier': 'SVC', 'svc_c': 4.877734600193295}

# Trial 51 finished with value: 0.9866666666666667 and parameters: {'classifier': 'SVC', 'svc_c': 4.877734600193295}. Best is trial 51 with value: 0.9866666666666667.
