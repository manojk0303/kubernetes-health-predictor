# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectFromModel
import joblib
import datetime
import logging
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kubernetes_prediction.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class KubernetesPredictionModel:
    """
    A class to predict issues in Kubernetes clusters based on metrics and logs.
    """
    
    def __init__(self, data_path=None, model_path=None):
        """
        Initialize the prediction model.
        
        Args:
            data_path (str): Path to the training data CSV file
            model_path (str): Path to save/load the trained model
        """
        self.data_path = data_path
        self.model_path = model_path
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        logger.info("Initialized Kubernetes Prediction Model")
    
    def load_data(self, data_path=None):
        """
        Load data from a CSV file.
        
        Args:
            data_path (str): Path to the data file (overrides init path if provided)
        
        Returns:
            pandas.DataFrame: The loaded data
        """
        if data_path:
            self.data_path = data_path
        
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully with shape: {self.data.shape}")
            # After loading the data
            missing_values = self.data.isnull().sum()
            logger.info(f"Missing values per column:\n{missing_values}")
            logger.info("Cleaning data and handling missing values")
            self.data = self.data.dropna()

            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, target_column='issue_type', test_size=0.25, random_state=42):
        """
        Preprocess the data for model training.
        
        Args:
            target_column (str): The column name for the target variable
            test_size (float): Proportion of the dataset to include in the test split
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            logger.error("No data loaded. Please load data first.")
            return None
        
        logger.info("Starting data preprocessing")
        
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Save feature names for later interpretation
        self.feature_names = X.columns.tolist()
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numeric features: {len(numeric_features)}, Categorical features: {len(categorical_features)}")
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')), 
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split into training set ({X_train.shape[0]} samples) and test set ({X_test.shape[0]} samples)")
        
        # Store for later use
        self.X = X
        self.y = y
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='random_forest'):
        """
        Train the prediction model.
        
        Args:
            X_train (pandas.DataFrame): Training features
            y_train (pandas.Series): Training target
            model_type (str): Type of model to train ('random_forest' or 'gradient_boosting')
            
        Returns:
            object: Trained model
        """
        logger.info(f"Training a {model_type} model")
        
        if model_type == 'random_forest':
            model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=None, 
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])
            
            # Train model
            logger.info("Fitting Random Forest model")
            model.fit(X_train, y_train)
            
        elif model_type == 'gradient_boosting':
            model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                ))
            ])
            
            # Train model
            logger.info("Fitting Gradient Boosting model")
            model.fit(X_train, y_train)
            
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model = model
        logger.info("Model training completed")
        
        return model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test (pandas.DataFrame): Test features
            y_test (pandas.Series): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            logger.error("No model trained. Please train a model first.")
            return None
        
        logger.info("Evaluating model performance")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        # Calculate evaluation metrics
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # For multi-class, calculate ROC AUC using one-vs-rest
        try:
            auc_score = roc_auc_score(pd.get_dummies(y_test), y_prob, multi_class='ovr')
        except:
            auc_score = None
            logger.warning("Could not calculate ROC AUC score")
        
        # Compile results
        evaluation = {
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'auc_score': auc_score
        }
        
        # Log key metrics
        logger.info(f"Accuracy: {class_report['accuracy']:.4f}")
        logger.info(f"Weighted F1: {class_report['weighted avg']['f1-score']:.4f}")
        if auc_score:
            logger.info(f"ROC AUC: {auc_score:.4f}")
        
        return evaluation
    
    def feature_importance(self):
        """
        Calculate and visualize feature importances.
        
        Returns:
            pandas.DataFrame: Feature importances
        """
        if self.model is None:
            logger.error("No model trained. Please train a model first.")
            return None
        
        logger.info("Calculating feature importances")
        
        # Extract the classifier from the pipeline
        classifier = self.model.named_steps['classifier']
        
        # Get feature importances
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            
            # Get feature names after preprocessing
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
            else:
                # Fallback for older scikit-learn versions
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            # Create DataFrame with feature importances
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info(f"Top 5 important features: {feature_importance_df['feature'].head(5).tolist()}")
            
            return feature_importance_df
        else:
            logger.warning("Model does not provide feature importances")
            return None
    
    def save_model(self, model_path=None):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path to save the model (overrides init path if provided)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No model to save. Please train a model first.")
            return False
        
        if model_path:
            self.model_path = model_path
        
        try:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path=None):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the model file (overrides init path if provided)
            
        Returns:
            object: Loaded model
        """
        if model_path:
            self.model_path = model_path
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def predict(self, X_new):
        """
        Make predictions on new data.
        
        Args:
            X_new (pandas.DataFrame): New data to predict on
            
        Returns:
            numpy.ndarray: Predicted classes and probabilities
        """
        if self.model is None:
            logger.error("No model loaded. Please train or load a model first.")
            return None
        
        logger.info(f"Making predictions on {X_new.shape[0]} samples")
        
        try:
            # Make predictions
            predictions = {
                'class': self.model.predict(X_new),
                'probabilities': self.model.predict_proba(X_new)
            }
            
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def visualize_results(self, evaluation=None):
        """
        Visualize model evaluation results.
        
        Args:
            evaluation (dict): Evaluation metrics from evaluate_model()
            
        Returns:
            tuple: Figure objects for visualizations
        """
        if not evaluation:
            logger.warning("No evaluation results provided.")
            return None
        
        logger.info("Creating visualizations for model evaluation")
        
        # Create confusion matrix heatmap
        plt.figure(figsize=(10, 8))
        conf_matrix = evaluation['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        conf_matrix_fig = plt.gcf()
        
        # Create feature importance plot if available
        importance_df = self.feature_importance()
        if importance_df is not None:
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title('Top 15 Feature Importances')
            plt.tight_layout()
            feature_fig = plt.gcf()
        else:
            feature_fig = None
        
        return conf_matrix_fig, feature_fig

# Example usage
if __name__ == "__main__":
    # Initialize the model
    k8s_model = KubernetesPredictionModel(
        data_path="kubernetes_metrics_dataset.csv",
        model_path="kubernetes_prediction_model.joblib"
    )
    
    # Load and preprocess data
    data = k8s_model.load_data()
    X_train, X_test, y_train, y_test = k8s_model.preprocess_data(target_column='issue_type')
    
    # Train model
    model = k8s_model.train_model(X_train, y_train, model_type='random_forest')
    
    # Evaluate model
    evaluation = k8s_model.evaluate_model(X_test, y_test)
    
    # Visualize results
    conf_matrix_fig, feature_fig = k8s_model.visualize_results(evaluation)
    
    # Save plots
    conf_matrix_fig.savefig('confusion_matrix.png')
    if feature_fig:
        feature_fig.savefig('feature_importance.png')
    
    # Save model
    k8s_model.save_model()
    
    print("Model training and evaluation completed!")