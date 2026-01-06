"""
Core Machine Learning Engine
Handles model training, evaluation, and comparison with automated explanations
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
import warnings
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import json

warnings.filterwarnings('ignore')

class MLEngine:
    def __init__(self, dataframe: pd.DataFrame, target_column: str, problem_type: str):
        """
        Initialize ML Engine with dataset
        
        Args:
            dataframe: Input DataFrame
            target_column: Target variable column name
            problem_type: Type of ML problem ('classification', 'regression', 'clustering')
        """
        self.df = dataframe.copy()
        self.target_column = target_column
        self.problem_type = problem_type
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.X_train_scaled: Optional[np.ndarray] = None
        self.X_test_scaled: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.models: Dict[str, Any] = {}
        
        # Prepare data
        self._prepare_data()
    
    def _safe_float_conversion(self, value: Any) -> float:
        """Safely convert any value to float"""
        try:
            if pd.isna(value):
                return 0.0
            if hasattr(value, 'item'):
                return float(value.item())
            return float(value)
        except (ValueError, TypeError, AttributeError):
            return 0.0
    
    def _prepare_data(self):
        """Prepare data for ML training"""
        # Separate features and target
        if self.target_column in self.df.columns:
            self.X = self.df.drop(columns=[self.target_column])
            self.y = self.df[self.target_column]
        else:
            self.X = self.df.copy()
            self.y = None
        
        # Handle categorical variables
        self._encode_categorical()
        
        # Handle missing values
        self._handle_missing_values()
    
    def _encode_categorical(self):
        """Encode categorical variables"""
        # FIX: Check if X is None
        if self.X is None:
            return
        
        # Get categorical columns safely
        cat_cols = self.X.select_dtypes(include=['object', 'category']).columns
        
        for column in cat_cols:
            le = LabelEncoder()
            self.X[column] = le.fit_transform(self.X[column].astype(str))
            self.label_encoders[column] = le
        
        # Encode target for classification
        if self.problem_type == 'classification' and self.y is not None:
            # FIX: Use hasattr to check dtype properly
            if hasattr(self.y, 'dtype') and self.y.dtype in ['object', 'category']:
                le_target = LabelEncoder()
                self.y = pd.Series(le_target.fit_transform(self.y))
                self.label_encoders['target'] = le_target
    
    def _handle_missing_values(self):
        """Handle missing values in features"""
        # FIX: Check if X is None
        if self.X is None:
            return
        
        # For numerical columns, fill with median
        num_cols = self.X.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.X[col].isnull().any():
                median_val = self.X[col].median()
                self.X[col].fillna(median_val, inplace=True)
        
        # For categorical columns, fill with mode
        cat_cols = self.X.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if self.X[col].isnull().any():
                mode_val = self.X[col].mode()
                if len(mode_val) > 0:
                    self.X[col].fillna(mode_val[0], inplace=True)
    
    def analyze_for_ml(self) -> Dict[str, Any]:
        """
        Analyze dataset for ML readiness
        Returns insights and recommendations
        """
        # FIX: Check if X is None
        if self.X is None:
            return {
                "error": "Features not prepared",
                "dataset_shape": self.df.shape,
                "target_column": self.target_column,
                "problem_type": self.problem_type
            }
        
        analysis = {
            "dataset_shape": self.df.shape,
            "target_column": self.target_column,
            "problem_type": self.problem_type,
            "features_count": len(self.X.columns),
            "categorical_features": list(self.X.select_dtypes(exclude=[np.number]).columns),
            "numerical_features": list(self.X.select_dtypes(include=[np.number]).columns),
            "issues": [],
            "recommendations": [],
            "model_recommendations": []
        }
        
        # Check for issues
        if self.y is not None:
            if self.problem_type == 'classification':
                class_distribution = pd.Series(self.y).value_counts()
                analysis["class_distribution"] = class_distribution.to_dict()
                
                # Check for class imbalance
                if len(class_distribution) > 0:
                    # FIX: Use safe float conversion
                    max_val = self._safe_float_conversion(class_distribution.max())
                    min_val = self._safe_float_conversion(class_distribution.min())
                    if min_val > 0:
                        imbalance_ratio = max_val / min_val
                        if imbalance_ratio > 10:
                            analysis["issues"].append("Severe class imbalance detected")
                            analysis["recommendations"].append("Consider using class_weight='balanced' or SMOTE")
            
            elif self.problem_type == 'regression':
                # FIX: Use numpy for type safety
                y_array = np.array(self.y, dtype=float)
                analysis["target_statistics"] = {
                    "mean": float(np.mean(y_array)),
                    "std": float(np.std(y_array)),
                    "min": float(np.min(y_array)),
                    "max": float(np.max(y_array))
                }
        
        # Check for multicollinearity - FIX: Null check
        if self.X is not None and len(self.X.columns) > 1:
            corr_matrix = self.X.corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    # FIX: Use safe float conversion
                    if pd.notna(corr_val):
                        corr_float = self._safe_float_conversion(corr_val)
                        if abs(corr_float) > 0.8:
                            high_corr_pairs.append({
                                "feature1": corr_matrix.columns[i],
                                "feature2": corr_matrix.columns[j],
                                "correlation": corr_float
                            })
            
            if high_corr_pairs:
                analysis["issues"].append("High correlation between features detected")
                analysis["high_correlation_pairs"] = high_corr_pairs[:5]
                analysis["recommendations"].append("Consider removing highly correlated features")
        
        # Recommend models based on problem type and data size - FIX: Null check
        n_samples = len(self.X) if self.X is not None else 0
        
        if self.problem_type == 'classification':
            if n_samples < 1000:
                analysis["model_recommendations"] = ["Logistic Regression", "Decision Tree", "Random Forest"]
            else:
                analysis["model_recommendations"] = ["Random Forest", "XGBoost", "SVM"]
        
        elif self.problem_type == 'regression':
            if n_samples < 1000:
                analysis["model_recommendations"] = ["Linear Regression", "Decision Tree", "Random Forest"]
            else:
                analysis["model_recommendations"] = ["Random Forest", "XGBoost", "SVR"]
        
        elif self.problem_type == 'clustering':
            analysis["model_recommendations"] = ["KMeans", "DBSCAN"]
        
        return analysis
    
    def train_model(self, model_type: str, test_size: float = 0.2, 
                   random_state: int = 42, **hyperparams) -> Dict[str, Any]:
        """
        Train a machine learning model
        
        Returns:
            Dictionary containing model, metrics, and training info
        """
        # FIX: Check if X is None
        if self.X is None:
            return {
                "error": "Features not prepared",
                "model_type": model_type
            }
        
        # Split data
        if self.y is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )
        else:
            # For clustering, use all data
            self.X_train = self.X.copy()
            self.X_test = None
            self.y_train = None
            self.y_test = None
        
        # Scale features
        self.scaler = StandardScaler()
        # FIX: Check for None and use proper type
        if self.X_train is not None:
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        else:
            return {"error": "Training data not available"}
        
        if self.X_test is not None:
            self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Select model
        model = self._get_model(model_type, hyperparams)
        
        # Train model
        start_time = time.time()
        
        if self.problem_type == 'clustering':
            model.fit(self.X_train_scaled)
            training_time = time.time() - start_time
            
            # For clustering, calculate inertia
            metrics = {
                "inertia": float(model.inertia_) if hasattr(model, 'inertia_') else None,
                "n_clusters": len(set(model.labels_)) if hasattr(model, 'labels_') else None
            }
            predictions = model.labels_
            
        else:
            model.fit(self.X_train_scaled, self.y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            if self.X_test_scaled is not None and self.y_test is not None:
                y_pred = model.predict(self.X_test_scaled)
                predictions = y_pred
                
                # Calculate metrics
                metrics = self._calculate_metrics(self.y_test, y_pred)
            else:
                return {"error": "Test data not available"}
        
        # Feature importance for tree-based models
        feature_importance = None
        if hasattr(model, 'feature_importances_') and self.X is not None:
            # FIX: Ensure feature_importances_ is converted properly
            importances = np.array(model.feature_importances_)
            importance_dict = dict(zip(self.X.columns, importances))
            feature_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Store model
        self.models[model_type] = model
        
        result = {
            "model": model,
            "model_type": model_type,
            "metrics": metrics,
            "training_time": training_time,
            "feature_importance": feature_importance,
            "predictions": predictions,
            "test_size": test_size,
            "hyperparameters": hyperparams
        }
        
        return result
    
    def _get_model(self, model_type: str, hyperparams: Dict) -> Any:
        """Get model instance based on type"""
        # Default hyperparameters
        default_hyperparams = {
            'random_state': 42,
            'n_estimators': 100
        }
        
        # Merge with provided hyperparams
        final_hyperparams = {**default_hyperparams, **hyperparams}
        
        model_map = {
            'classification': {
                'logistic_regression': LogisticRegression,
                'random_forest': RandomForestClassifier,
                'decision_tree': DecisionTreeClassifier,
                'svm': SVC,
                'xgboost': XGBClassifier,
                'naive_bayes': GaussianNB
            },
            'regression': {
                'linear_regression': LinearRegression,
                'random_forest': RandomForestRegressor,
                'decision_tree': DecisionTreeRegressor,
                'svr': SVR,
                'xgboost': XGBRegressor
            },
            'clustering': {
                'kmeans': KMeans,
                'dbscan': DBSCAN
            }
        }
        
        model_key = model_type.lower().replace(" ", "_")
        model_class = model_map.get(self.problem_type, {}).get(
            model_key, 
            model_map[self.problem_type].get('random_forest', RandomForestClassifier)
        )
        
        # Handle models that don't accept certain parameters
        try:
            if model_key == 'linear_regression':
                return model_class()
            elif model_key == 'naive_bayes':
                return model_class()
            else:
                return model_class(**final_hyperparams)
        except Exception as e:
            # Fallback to default model
            return model_class()
    
    def _calculate_metrics(self, y_true: Union[pd.Series, np.ndarray], 
                          y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate appropriate metrics based on problem type"""
        # Convert to numpy arrays for type safety
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        
        if self.problem_type == 'classification':
            return {
                "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
                "precision": float(precision_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0))
            }
        elif self.problem_type == 'regression':
            mse = mean_squared_error(y_true_arr, y_pred_arr)
            return {
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(y_true_arr, y_pred_arr)),
                "mae": float(np.mean(np.abs(y_true_arr - y_pred_arr)))
            }
        return {}
    
    def compare_models(self, model_types: List[str]) -> Dict[str, Any]:
        """Compare multiple models side by side"""
        comparison = {
            "models": [],
            "best_model": None,
            "best_score": -float('inf')
        }
        
        for model_type in model_types:
            try:
                result = self.train_model(model_type, test_size=0.2, random_state=42)
                
                if "error" in result:
                    continue
                
                model_info = {
                    "model_type": model_type,
                    "metrics": result["metrics"],
                    "training_time": result["training_time"],
                    "hyperparameters": result.get("hyperparameters", {})
                }
                
                comparison["models"].append(model_info)
                
                # Determine best model based on primary metric
                score_key = "accuracy" if self.problem_type == "classification" else "r2"
                current_score = result["metrics"].get(score_key, -float('inf'))
                
                if current_score > comparison["best_score"]:
                    comparison["best_score"] = float(current_score)
                    comparison["best_model"] = model_type
                    comparison["best_model_reason"] = f"Highest {score_key}: {current_score:.3f}"
                    
            except Exception as e:
                print(f"Error training {model_type}: {str(e)}")
                continue
        
        return comparison
    
    def predict(self, model: Any, new_data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        # Preprocess new data
        new_data_processed = new_data.copy()
        
        # Apply same preprocessing
        for column, encoder in self.label_encoders.items():
            if column in new_data_processed.columns and column != 'target':
                try:
                    new_data_processed[column] = encoder.transform(new_data_processed[column].astype(str))
                except Exception:
                    # Handle unseen labels
                    new_data_processed[column] = encoder.transform(
                        new_data_processed[column].astype(str).apply(
                            lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                        )
                    )
        
        # Scale features
        if self.scaler:
            new_data_scaled = self.scaler.transform(new_data_processed)
        else:
            new_data_scaled = new_data_processed.values
        
        # Make predictions
        predictions = model.predict(new_data_scaled)
        
        return predictions
    
    def get_test_data(self) -> Optional[pd.DataFrame]:
        """Get test data for example predictions"""
        if self.X_test is not None:
            test_df = self.X_test.copy()
            if self.y_test is not None:
                test_df[self.target_column] = self.y_test
            return test_df
        return None
    
    def get_model_options(self, model_type: str) -> Dict[str, Any]:
        """Get configuration options for a model type"""
        options = {
            "hyperparameters": {},
            "recommendations": [],
            "considerations": []
        }
        
        if model_type == "Random Forest":
            options["hyperparameters"] = {
                "n_estimators": "Number of trees (default: 100)",
                "max_depth": "Maximum tree depth (default: None)",
                "min_samples_split": "Min samples to split (default: 2)"
            }
            options["recommendations"] = [
                "Use n_estimators=100-200 for good performance",
                "Set max_depth to prevent overfitting"
            ]
        
        elif model_type == "Logistic Regression":
            options["recommendations"] = [
                "Good for binary classification problems",
                "Fast and interpretable"
            ]
        
        elif model_type == "XGBoost":
            options["hyperparameters"] = {
                "n_estimators": "Number of boosting rounds (default: 100)",
                "learning_rate": "Step size (default: 0.3)",
                "max_depth": "Maximum tree depth (default: 6)"
            }
            options["recommendations"] = [
                "Best for competition-level accuracy",
                "Handles missing values automatically"
            ]
        
        return options