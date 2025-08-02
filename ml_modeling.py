import pandas as pd
import numpy as np
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

data_path = ''
predicted_col = ''
df = pd.read_csv(data_path, usecols=['Standard Formula', predicted_col, 'Material type', 'doi'])

df = df[pd.to_numeric(df[predicted_col], errors='coerce').notna()]
df[predicted_col] = df[predicted_col].astype(float)
df = df[df['Standard Formula'].notna()]

if 'Material type' in df.columns:
    df = df[df['Material type'] == 'Interstitial Hydride']

def safe_composition(x):
    try:
        return Composition(x)
    except Exception:
        return np.nan

df['composition'] = df['Standard Formula'].apply(safe_composition)
df = df[df['composition'].notna()]

featurizer = ElementProperty.from_preset('magpie', impute_nan=True)
X = featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)

def get_element_fractions(comp):
    if not comp or comp is np.nan:
        return {}
    d = comp.get_el_amt_dict()
    total = sum(d.values())
    return {k: v/total for k, v in d.items()}

df['element_fractions'] = df['composition'].apply(get_element_fractions)
all_elements = set()
for d in df['element_fractions']:
    all_elements.update(d.keys())
for el in sorted(all_elements):
    X[f'frac_{el}'] = df['element_fractions'].apply(lambda x: x.get(el, 0))

y = df[predicted_col]
X = X.drop([predicted_col, 'element_fractions', 'Material type'], axis=1, errors='ignore')
X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 5]
}
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, tree_method="hist")
grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
print('Best parameters:', grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.4f}, R2: {r2:.4f}')

best_model.save_model('xgb_model.json')
with open('feature_names.txt', 'w') as f:
    for col in X.columns.tolist():
        f.write(col + '\n')

def predict_formula(formula, model_path='xgb_model.json', feature_file='feature_names.txt'):
    model = XGBRegressor()
    model.load_model(model_path)
    with open(feature_file, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    comp = Composition(formula)
    df_new = pd.DataFrame({'composition': [comp]})
    featurizer = ElementProperty.from_preset('magpie', impute_nan=True)
    X_new = featurizer.featurize_dataframe(df_new, 'composition', ignore_errors=True)
    def get_element_fractions(comp):
        if not comp or comp is np.nan:
            return {}
        d = comp.get_el_amt_dict()
        total = sum(d.values())
        return {k: v/total for k, v in d.items()}
    element_fractions = get_element_fractions(comp)
    for el in [col.replace('frac_', '') for col in feature_names if col.startswith('frac_')]:
        X_new[f'frac_{el}'] = element_fractions.get(el, 0)
    X_new = X_new[feature_names]
    y_pred_new = model.predict(X_new)
    return y_pred_new[0]

result = predict_formula("Ca1Mg1Ni4")
print("Predicted hydrogen density for Ca1Mg1Ni4:", result)
