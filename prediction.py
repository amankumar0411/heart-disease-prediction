
import joblib, pandas as pd, numpy as np
from sklearn.impute import SimpleImputer

class HeartDiseasePredictor:
    def __init__(self):
        self.models = {
            'Logistic Regression': joblib.load("models/logistic_regression_model.pkl"),
            'SVM': joblib.load("models/svm_model.pkl"),
            'KNN': joblib.load("models/knn_model.pkl"),
            'Random Forest': joblib.load("models/random_forest_model.pkl"),
        }
        self.pre = joblib.load("models/preprocessor.pkl")
        self.feature_names = joblib.load("models/feature_names.pkl")
        self.scaler = self.pre['scaler']
        self.imputer = SimpleImputer(strategy='median')

    def _enforce_schema(self, df_row):
        for c in self.feature_names:
            if c not in df_row.columns:
                df_row[c] = np.nan
        return df_row[self.feature_names]

    def prepare_input(self, input_data):
        if isinstance(input_data, dict):
            df_row = pd.DataFrame([input_data])
        else:
            df_row = pd.DataFrame(input_data, columns=self.feature_names)
        df_row = self._enforce_schema(df_row)
        df_row = df_row.replace(['', ' ', 'NA', 'NaN', 'nan', '?'], np.nan)
        for c in ['age','trestbps','chol','thalach','oldpeak','ca']:
            if c in df_row.columns:
                df_row[c] = pd.to_numeric(df_row[c], errors='coerce')
        X_imp = self.imputer.fit_transform(df_row.values)
        X_sc  = self.scaler.transform(X_imp)
        return X_sc

    def predict_single(self, input_dict, model_name='SVM'):
        X_sc = self.prepare_input(input_dict)
        model = self.models[model_name]
        pred = int(model.predict(X_sc))
        prob = float(model.predict_proba(X_sc)[0,1])
        risk = "Low Risk" if prob < 0.3 else ("Moderate Risk" if prob < 0.7 else "High Risk")
        return {'prediction': pred, 'probability_disease': prob, 'risk_level': risk}
