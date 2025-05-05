import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import seaborn as sns
from typing import List, Tuple, Dict, Any
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDiagnosisSystem:
    def __init__(self, data_path: str):
        """Initialize the Medical Diagnosis System."""
        self.data_path = data_path
        self.disease_mapping = None
        self.feature_columns = None
        self.best_model = None
        
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess the medical dataset."""
        try:
            if not os.path.exists(self.data_path):
                st.error(f"Data file not found: {self.data_path}")
                st.info("Please make sure your dataset file is in the correct location.")
                st.stop()
                
            # Load data
            df = pd.read_csv(self.data_path)
            
            # Display data info
            st.write("Dataset Info:")
            st.write(f"Total samples: {len(df)}")
            st.write(f"Total features: {len(df.columns)}")
            
            # Handle missing values
            df.fillna("No Symptom", inplace=True)
            
            # Convert categorical symptoms into numerical using One-Hot Encoding
            symptom_columns = [col for col in df.columns if col != "Disease"]
            
            # Create disease mapping before encoding features
            unique_diseases = df["Disease"].unique()
            self.disease_mapping = {disease: idx for idx, disease in enumerate(unique_diseases)}
            
            # Convert to numeric features
            df_encoded = pd.get_dummies(df[symptom_columns], dtype=np.int8)
            self.feature_columns = list(df_encoded.columns)
            
            # Convert labels to numeric
            y = df["Disease"].map(self.disease_mapping).astype(np.int32)
            
            # Display encoding info
            st.write(f"Encoded features: {len(self.feature_columns)}")
            st.write(f"Unique diseases: {len(self.disease_mapping)}")
            
            return df_encoded, y
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            st.error(f"Error processing data: {str(e)}")
            st.stop()

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train and evaluate multiple models."""
        try:
            # Convert data to float32 for better compatibility
            X = X.astype(np.float32)
            y = y.astype(np.int32)
            
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Apply SMOTE
            st.write("Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            # Convert to proper types after SMOTE
            X_train_balanced = pd.DataFrame(X_train_balanced, columns=X.columns).astype(np.float32)
            y_train_balanced = pd.Series(y_train_balanced).astype(np.int32)
            
            # Train Random Forest
            st.write("Training Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_balanced, y_train_balanced)
            
            # Train XGBoost
            st.write("Training XGBoost...")
            xgb_model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=7,
                random_state=42,
                eval_metric='mlogloss'
            )
            
            # Simple fit without early stopping
            xgb_model.fit(X_train_balanced, y_train_balanced)
            
            # Evaluate models
            models = {
                "Random Forest": rf_model,
                "XGBoost": xgb_model
            }
            
            results = {}
            for name, model in models.items():
                y_pred = model.predict(X_test)
                f1 = f1_score(y_test, y_pred, average='weighted')
                results[name] = f1
                
                # Generate detailed metrics
                st.write(f"\nModel: {name}")
                st.write(f"F1 Score: {f1:.4f}")
                
                # Plot confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d')
                plt.title(f'Confusion Matrix - {name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                st.pyplot(plt)
                
            # Select best model
            best_model_name = max(results, key=results.get)
            self.best_model = models[best_model_name]
            st.success(f"Best model: {best_model_name} (F1 Score: {results[best_model_name]:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            st.error(f"Error during model training: {str(e)}")
            st.stop()

    def save_model(self, filepath: str):
        """Save the trained model and necessary mappings."""
        try:
            with open(filepath, "wb") as file:
                pickle.dump({
                    'model': self.best_model,
                    'disease_mapping': self.disease_mapping,
                    'feature_columns': self.feature_columns
                }, file)
            logger.info(f"Model saved successfully to {filepath}")
            st.success("Model saved successfully!")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            st.error(f"Error saving model: {str(e)}")
            st.stop()

    def predict_disease(self, symptoms: List[str]) -> List[Tuple[str, float]]:
        """Predict diseases based on input symptoms."""
        try:
            # Validate symptoms
            unknown_symptoms = [symptom for symptom in symptoms 
                              if symptom not in self.feature_columns]
            if unknown_symptoms:
                st.warning(f"Unknown symptoms: {', '.join(unknown_symptoms)}")
            
            # Prepare input data
            input_data = np.zeros(len(self.feature_columns), dtype=np.float32)
            for symptom in symptoms:
                if symptom in self.feature_columns:
                    input_data[self.feature_columns.index(symptom)] = 1
            
            # Get predictions and probabilities
            probabilities = self.best_model.predict_proba([input_data])[0]
            disease_probs = sorted(zip(self.best_model.classes_, probabilities), 
                                 key=lambda x: x[1], reverse=True)
            
            # Decode predictions
            reverse_mapping = {v: k for k, v in self.disease_mapping.items()}
            decoded_predictions = [(reverse_mapping[d], prob) 
                                 for d, prob in disease_probs]
            
            return decoded_predictions[:3]
            
        except Exception as e:
            logger.error(f"Error in disease prediction: {str(e)}")
            st.error(f"Error during prediction: {str(e)}")
            st.stop()

def main():
    st.title("Advanced Medical Diagnosis AI")
    
    # Initialize system
    system = MedicalDiagnosisSystem("medical_dataset_20250216_214908.csv")
    
    # Check if model exists
    model_exists = os.path.exists("medical_diagnosis_model.pkl")
    
    # Load model and feature columns if exists
    if model_exists and not system.feature_columns:
        with open("medical_diagnosis_model.pkl", "rb") as file:
            data = pickle.load(file)
            system.best_model = data['model']
            system.disease_mapping = data['disease_mapping']
            system.feature_columns = data['feature_columns']
    
    # First-time setup instructions
    if not model_exists:
        st.warning("No trained model found. Please train a new model first.")
        st.info("Click 'Train New Model' in the sidebar to begin.")
    
    # Sidebar options
    with st.sidebar:
        st.title("Settings")
        if st.checkbox("Train New Model", value=not model_exists):
            st.subheader("Model Training")
            if st.button("Start Training"):
                with st.spinner("Training models..."):
                    try:
                        X, y = system.load_and_preprocess_data()
                        results = system.train_models(X, y)
                        system.save_model("medical_diagnosis_model.pkl")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
        
        # Add symptom display in sidebar
        if system.feature_columns:
            st.subheader("Available Symptoms")
            st.write("Valid symptoms you can use:")
            # Display symptoms in a scrollable container
            # with st.container():
            #     for symptom in sorted(system.feature_columns):
            #         st.write(f"- {symptom}")
    
    # Main interface
    st.write("Enter symptoms to predict possible diseases.")
    st.write("Please separate multiple symptoms with commas.")
    
    # Example usage
    st.info("Example: fever, cough, headache")
    
    # Prediction interface
    user_symptoms = st.text_input("Enter symptoms:")
    
    if user_symptoms:
        if not model_exists:
            st.error("Please train a model first using the sidebar option.")
            st.stop()
            
        try:
            # Load saved model if not in training mode
            if not system.best_model:
                with open("medical_diagnosis_model.pkl", "rb") as file:
                    data = pickle.load(file)
                    system.best_model = data['model']
                    system.disease_mapping = data['disease_mapping']
                    system.feature_columns = data['feature_columns']
            
            user_symptoms_list = [symptom.strip().lower() for symptom in user_symptoms.split(",")]
            predictions = system.predict_disease(user_symptoms_list)
            
            st.subheader("Top Predicted Diseases:")
            for disease, prob in predictions:
                st.write(f"{disease}: {prob * 100:.2f}% confidence")
                
                # Add confidence level indicator
                confidence_color = "red" if prob < 0.3 else "yellow" if prob < 0.7 else "green"
                st.markdown(f"""
                    <div style='background-color: {confidence_color}; 
                               width: {prob * 100}%; 
                               height: 10px; 
                               border-radius: 5px;'></div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            if "medical_diagnosis_model.pkl" in str(e):
                st.info("The model file appears to be missing or corrupted. Please retrain the model.")

if __name__ == "__main__":
    main()