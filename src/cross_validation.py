from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    
def cross_validation_regression(model, X, y, folds=5, name="", model_name=""):
    
    def smape(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    
    def write_cv_results_to_file(cv_results, model_name, name):
        output_path = Path(f"graphs/models/{model_name}/{name}/cv_results.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for metric_name, metric_values in cv_results.items():
                if metric_name.startswith('test_'):
                    # Formatear el nombre de la métrica para mejorar la legibilidad
                    formatted_name = metric_name[5:].replace('_', ' ').capitalize()
                    # Calcular promedio y desviación estándar
                    mean_value = np.mean(metric_values)
                    std_dev = np.std(metric_values)
                    # Ajustar el formato del mensaje según el tipo de métrica
                    if "mape" in metric_name.lower() or "smape" in metric_name.lower():
                        f.write(f"{formatted_name}: {mean_value:.2f}% (+-{std_dev:.2f}%)\n")
                    else:
                        f.write(f"{formatted_name}: {mean_value:.4f} (+-{std_dev:.4f})\n")

    smape_scorer = make_scorer(smape, greater_is_better=False)  # Make smape compatible

    scoring = {
        "MSE": make_scorer(mean_squared_error, greater_is_better=False),
        "RMSE": make_scorer(
            lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            greater_is_better=False,
        ),
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "MAPE": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        "SMAPE": smape_scorer,
        "R2": "r2",
    }
    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(model, X, y, cv=folds, scoring=scoring)

    write_cv_results_to_file(cv_results, model_name, name)