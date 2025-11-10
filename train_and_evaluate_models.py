import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import warnings
import time
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# CONFIGURATION
CONFIG = {
    'data': {
        'file_path': 'stress_data.csv',
        'test_size': 0.2,
        'random_state': RANDOM_SEED,
        'stratify': True
    },
    'features': {
        'columns': ['Age', 'Heart_Rate', 'Pulse_Rate', 
                   'Sleep_Hours', 'Sleep_Quality', 'Physical_Activity'],
        'target': 'Stress_Category'
    },
    'models': {
        'RandomForest': {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': RANDOM_SEED,
            'class_weight': 'balanced',
            'n_jobs': -1
        },
        'GradientBoosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': RANDOM_SEED
        },
        'SVM_RBF': {
            'kernel': 'rbf',
            'C': 10,
            'gamma': 'scale',
            'probability': True,
            'random_state': RANDOM_SEED,
            'class_weight': 'balanced'
        },
        'SVM_Linear': {
            'kernel': 'linear',
            'C': 1.0,
            'probability': True,
            'random_state': RANDOM_SEED,
            'class_weight': 'balanced'
        },
        'DecisionTree': {
            'max_depth': 20,
            'min_samples_split': 3,
            'random_state': RANDOM_SEED,
            'class_weight': 'balanced'
        },
        'KNN': {
            'n_neighbors': 5,
            'weights': 'distance',
            'n_jobs': -1
        },
        'LogisticRegression': {
            'max_iter': 1000,
            'random_state': RANDOM_SEED,
            'class_weight': 'balanced'
        }
    },
    'evaluation': {
        'cv_folds': 5,
        'scoring': 'accuracy'
    },
    'output': {
        'model_dir': 'models',
        'results_dir': 'results',
        'figures_dir': 'figures'
    }
}

# UTILITY FUNCTIONS
def create_output_directories():
    """Create necessary output directories."""
    for dir_key in ['model_dir', 'results_dir', 'figures_dir']:
        Path(CONFIG['output'][dir_key]).mkdir(parents=True, exist_ok=True)

def save_config(filename='config.json'):
    """Save configuration to JSON file."""
    with open(filename, 'w') as f:
        json.dump(CONFIG, f, indent=4)
    print(f"✅ Configuration saved to {filename}")

# DATA LOADING AND PREPROCESSING
class DataProcessor:
    """Handle data loading, preprocessing, and splitting."""

    def __init__(self, config):
        self.config = config
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None

    def load_data(self):
        """Load and prepare dataset."""
        print("\n" + "="*80)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("="*80)

        try:
            self.df = pd.read_csv(self.config['data']['file_path'])
            print(f"✅ Dataset loaded: {self.df.shape}")
        except FileNotFoundError:
            # Try alternative filename
            self.df = pd.read_csv('student_stress_balanced.csv')
            print(f"✅ Dataset loaded: {self.df.shape}")

        # Create stress categories if needed
        if self.config['features']['target'] not in self.df.columns:
            self.df[self.config['features']['target']] = pd.cut(
                self.df['Stress_Level'],
                bins=[0, 3, 6, 10],
                labels=['Low', 'Medium', 'High']
            )

        # Display distribution
        print(f"\nClass distribution:")
        for cls in ['Low', 'Medium', 'High']:
            count = (self.df[self.config['features']['target']] == cls).sum()
            pct = count / len(self.df) * 100
            print(f"  {cls}: {count} ({pct:.1f}%)")

        return self.df

    def prepare_features(self):
        """Extract and preprocess features."""
        X = self.df[self.config['features']['columns']].copy()
        y = self.df[self.config['features']['target']].copy()

        # Handle missing values
        X.fillna(X.mean(), inplace=True)

        print(f"\nFeatures prepared: {X.shape}")
        print(f"Feature names: {list(X.columns)}")

        return X, y

    def split_data(self, X, y):
        """Split data into train and test sets."""
        stratify = y if self.config['data']['stratify'] else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=stratify
        )

        print(f"\nData split:")
        print(f"  Training set: {len(self.X_train)} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"  Test set: {len(self.X_test)} samples ({len(self.X_test)/len(X)*100:.1f}%)")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_features(self):
        """Scale features using StandardScaler."""
        print("\n" + "="*80)
        print("STEP 2: FEATURE SCALING")
        print("="*80)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        print(f"✅ Features scaled using StandardScaler")
        print(f"   Mean: {X_train_scaled.mean():.6f}")
        print(f"   Std: {X_train_scaled.std():.6f}")

        return X_train_scaled, X_test_scaled

# MODEL TRAINING AND EVALUATION
class ModelTrainer:
    """Train and evaluate multiple machine learning models."""

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.results = {}

    def get_model_instance(self, model_name, params):
        """Create model instance based on name and parameters."""
        model_mapping = {
            'RandomForest': RandomForestClassifier,
            'GradientBoosting': GradientBoostingClassifier,
            'SVM_RBF': SVC,
            'SVM_Linear': SVC,
            'DecisionTree': DecisionTreeClassifier,
            'KNN': KNeighborsClassifier,
            'LogisticRegression': LogisticRegression
        }

        return model_mapping[model_name](**params)

    def train_and_evaluate(self, X_train_scaled, X_test_scaled, y_train, y_test):
        """Train all models and evaluate performance."""
        print("\n" + "="*80)
        print("STEP 3: MODEL TRAINING AND EVALUATION")
        print("="*80)

        for model_name, params in self.config['models'].items():
            print(f"\n{'='*80}")
            print(f"Training: {model_name}")
            print(f"{'='*80}")

            # Create and train model
            model = self.get_model_instance(model_name, params)

            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_train = model.predict(X_train_scaled)

            # Metrics
            test_acc = accuracy_score(y_test, y_pred)
            train_acc = accuracy_score(y_train, y_pred_train)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Cross-validation
            cv = StratifiedKFold(n_splits=self.config['evaluation']['cv_folds'], 
                                shuffle=True, random_state=RANDOM_SEED)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                       cv=cv, scoring=self.config['evaluation']['scoring'])

            # Store results
            self.models[model_name] = model
            self.results[model_name] = {
                'accuracy': test_acc,
                'train_accuracy': train_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'y_pred': y_pred,
                'y_test': y_test,
                'params': params
            }

            # Print results
            print(f"  ✅ Training time: {training_time*1000:.2f} ms")
            print(f"  ✅ Test accuracy: {test_acc*100:.2f}%")
            print(f"  ✅ Train accuracy: {train_acc*100:.2f}%")
            print(f"  ✅ Precision: {precision:.4f}")
            print(f"  ✅ Recall: {recall:.4f}")
            print(f"  ✅ F1-Score: {f1:.4f}")
            print(f"  ✅ CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

            # Overfitting check
            overfit = train_acc - test_acc
            if overfit > 0.1:
                print(f"  ⚠️  Overfitting detected (diff: {overfit:.4f})")
            else:
                print(f"  ✓ Good generalization (diff: {overfit:.4f})")

        return self.models, self.results

    def print_comparison(self):
        """Print comprehensive model comparison."""
        print("\n" + "="*80)
        print("STEP 4: MODEL COMPARISON SUMMARY")
        print("="*80)

        # Sort by accuracy
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1]['accuracy'], reverse=True)

        # Table header
        print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*85)

        for model_name, metrics in sorted_results:
            print(f"{model_name:<25} {metrics['accuracy']:<12.4f} "
                  f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f}")

        print("-"*85)

        # Best model
        best_model_name = sorted_results[0][0]
        best_metrics = self.results[best_model_name]

        print(f"\n✅ BEST MODEL: {best_model_name}")
        print(f"   Accuracy: {best_metrics['accuracy']*100:.2f}%")
        print(f"   F1-Score: {best_metrics['f1_score']:.4f}")
        print(f"   CV Score: {best_metrics['cv_mean']:.4f} (±{best_metrics['cv_std']:.4f})")

        return best_model_name

    def save_best_model(self, scaler, feature_columns):
        """Save the best performing model and associated files."""
        print("\n" + "="*80)
        print("STEP 5: SAVING MODELS AND RESULTS")
        print("="*80)

        # Get best model
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1]['accuracy'], reverse=True)
        best_model_name = sorted_results[0][0]
        best_model = self.models[best_model_name]

        # Save model
        model_path = Path(self.config['output']['model_dir']) / 'best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"✅ Model saved: {model_path}")

        # Save scaler
        scaler_path = Path(self.config['output']['model_dir']) / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler saved: {scaler_path}")

        # Save feature names
        features_path = Path(self.config['output']['model_dir']) / 'features.json'
        with open(features_path, 'w') as f:
            json.dump({'features': feature_columns}, f)
        print(f"✅ Feature list saved: {features_path}")

        # Save model info
        info_path = Path(self.config['output']['results_dir']) / 'model_info.txt'
        with open(info_path, 'w') as f:
            f.write(f"STRESS LEVEL PREDICTION - MODEL INFORMATION\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Accuracy: {self.results[best_model_name]['accuracy']*100:.2f}%\n")
            f.write(f"F1-Score: {self.results[best_model_name]['f1_score']:.4f}\n")
            f.write(f"Features: {feature_columns}\n")
            f.write(f"Random Seed: {RANDOM_SEED}\n")
        print(f"✅ Model info saved: {info_path}")

        # Save all results
        results_path = Path(self.config['output']['results_dir']) / 'all_results.json'
        serializable_results = {}
        for model_name, metrics in self.results.items():
            serializable_results[model_name] = {
                k: v for k, v in metrics.items() 
                if k not in ['y_pred', 'y_test']
            }
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"✅ All results saved: {results_path}")

# MAIN EXECUTION
def main():
    """Main execution function."""
    print("\n" + "#"*80)
    print("#" + " "*20 + "STRESS LEVEL PREDICTION - MODEL TRAINING" + " "*19 + "#")
    print("#" + " "*25 + "Version 2.0 (Publication Ready)" + " "*22 + "#")
    print("#"*80)

    try:
        # Create output directories
        create_output_directories()

        # Save configuration
        save_config()

        # Data processing
        processor = DataProcessor(CONFIG)
        df = processor.load_data()
        X, y = processor.prepare_features()
        X_train, X_test, y_train, y_test = processor.split_data(X, y)
        X_train_scaled, X_test_scaled = processor.scale_features()

        # Model training
        trainer = ModelTrainer(CONFIG)
        models, results = trainer.train_and_evaluate(
            X_train_scaled, X_test_scaled, y_train, y_test
        )

        # Print comparison
        best_model_name = trainer.print_comparison()

        # Save results
        trainer.save_best_model(processor.scaler, CONFIG['features']['columns'])

        # Final message
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE")
        print("="*80)
        print(f"\nBest Model: {best_model_name}")
        print(f"Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
        print(f"\nOutput files created in:")
        print(f"  - {CONFIG['output']['model_dir']}/")
        print(f"  - {CONFIG['output']['results_dir']}/")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
