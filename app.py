"""
Modern Streamlit web UI for adversarial attack generation and visualization.
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image
import io
import base64
import json
import os
from datetime import datetime
import sqlite3
from typing import Dict, List, Tuple

# Import our modules
from models.cnn_model import create_model, load_pretrained_model
from attacks.adversarial_attacks import AdversarialAttacks, evaluate_attack_robustness
from train import Trainer


class AdversarialAttackUI:
    """
    Streamlit UI for adversarial attack generation and analysis.
    """
    
    def __init__(self):
        """Initialize the UI."""
        self.setup_page_config()
        self.setup_database()
        self.load_model()
    
    def setup_page_config(self):
        """Configure Streamlit page."""
        st.set_page_config(
            page_title="Adversarial Attack Generator",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .attack-success {
            color: #d62728;
            font-weight: bold;
        }
        .attack-failed {
            color: #2ca02c;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def setup_database(self):
        """Setup SQLite database for storing results."""
        self.db_path = "adversarial_attacks.db"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attack_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            attack_method TEXT,
            epsilon REAL,
            original_prediction INTEGER,
            adversarial_prediction INTEGER,
            true_label INTEGER,
            success BOOLEAN,
            confidence_original REAL,
            confidence_adversarial REAL,
            image_data BLOB
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_type TEXT,
            accuracy REAL,
            robustness_score REAL,
            training_epochs INTEGER
        )
        """)
        
        conn.commit()
        conn.close()
    
    def load_model(self):
        """Load the pre-trained model."""
        if 'model' not in st.session_state:
            try:
                # Try to load pre-trained model
                model_path = "./checkpoints/best_model.pth"
                if os.path.exists(model_path):
                    st.session_state.model = load_pretrained_model(
                        model_path, 'modern', 'cpu'
                    )
                    st.session_state.model_loaded = True
                else:
                    # Create a new model if no pre-trained model exists
                    st.session_state.model = create_model('modern').to('cpu')
                    st.session_state.model_loaded = False
                    st.warning("No pre-trained model found. Please train a model first.")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.session_state.model = create_model('modern').to('cpu')
                st.session_state.model_loaded = False
    
    def save_attack_result(self, attack_data: Dict):
        """Save attack result to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO attack_results 
        (attack_method, epsilon, original_prediction, adversarial_prediction, 
         true_label, success, confidence_original, confidence_adversarial, image_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            attack_data['method'],
            attack_data['epsilon'],
            attack_data['original_pred'],
            attack_data['adversarial_pred'],
            attack_data['true_label'],
            attack_data['success'],
            attack_data['confidence_original'],
            attack_data['confidence_adversarial'],
            attack_data['image_data']
        ))
        
        conn.commit()
        conn.close()
    
    def get_attack_statistics(self) -> Dict:
        """Get attack statistics from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall statistics
        cursor.execute("""
        SELECT 
            COUNT(*) as total_attacks,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_attacks,
            AVG(epsilon) as avg_epsilon,
            attack_method,
            COUNT(*) as method_count
        FROM attack_results 
        GROUP BY attack_method
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return {
            'total_attacks': sum([r[0] for r in results]),
            'successful_attacks': sum([r[1] for r in results]),
            'avg_epsilon': sum([r[2] for r in results]) / len(results) if results else 0,
            'method_stats': {r[3]: r[4] for r in results}
        }
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">🎯 Adversarial Attack Generator</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Generate and analyze adversarial examples using state-of-the-art attack methods
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.header("⚙️ Attack Configuration")
        
        # Attack method selection
        attack_method = st.sidebar.selectbox(
            "Attack Method",
            ["FGSM", "PGD", "BIM", "DeepFool", "C&W"],
            help="Choose the adversarial attack method"
        )
        
        # Epsilon value
        epsilon = st.sidebar.slider(
            "Epsilon (Perturbation Magnitude)",
            min_value=0.01,
            max_value=0.5,
            value=0.25,
            step=0.01,
            help="Controls the strength of the adversarial perturbation"
        )
        
        # Additional parameters based on attack method
        if attack_method == "PGD":
            num_iter = st.sidebar.slider("Number of Iterations", 1, 50, 10)
            alpha = st.sidebar.slider("Step Size (Alpha)", 0.001, 0.1, 0.01)
        elif attack_method == "BIM":
            num_iter = st.sidebar.slider("Number of Iterations", 1, 20, 10)
        elif attack_method == "C&W":
            c = st.sidebar.slider("Confidence Parameter (C)", 0.1, 10.0, 1.0)
        else:
            num_iter = None
            alpha = None
            c = None
        
        return {
            'method': attack_method.lower().replace('&', '_').replace(' ', '_'),
            'epsilon': epsilon,
            'num_iter': num_iter,
            'alpha': alpha,
            'c': c
        }
    
    def generate_adversarial_example(self, image: torch.Tensor, 
                                   label: torch.Tensor, 
                                   config: Dict) -> Tuple[torch.Tensor, Dict]:
        """Generate adversarial example."""
        attacks = AdversarialAttacks(st.session_state.model, 'cpu')
        
        method = config['method']
        epsilon = config['epsilon']
        
        # Get attack function
        if method == 'fgsm':
            adv_image = attacks.fgsm_attack(image, label, epsilon)
        elif method == 'pgd':
            adv_image = attacks.pgd_attack(image, label, epsilon, 
                                         config['alpha'], config['num_iter'])
        elif method == 'bim':
            adv_image = attacks.bim_attack(image, label, epsilon, 
                                          config['alpha'], config['num_iter'])
        elif method == 'deepfool':
            adv_image = attacks.deepfool_attack(image, label)
        elif method == 'c_w':
            adv_image = attacks.cw_attack(image, label, config['c'])
        else:
            raise ValueError(f"Unknown attack method: {method}")
        
        # Get predictions and confidence
        with torch.no_grad():
            original_output = st.session_state.model(image)
            adversarial_output = st.session_state.model(adv_image)
            
            original_pred = original_output.argmax(dim=1).item()
            adversarial_pred = adversarial_output.argmax(dim=1).item()
            
            original_confidence = F.softmax(original_output, dim=1).max().item()
            adversarial_confidence = F.softmax(adversarial_output, dim=1).max().item()
        
        attack_data = {
            'method': method,
            'epsilon': epsilon,
            'original_pred': original_pred,
            'adversarial_pred': adversarial_pred,
            'true_label': label.item(),
            'success': original_pred != adversarial_pred,
            'confidence_original': original_confidence,
            'confidence_adversarial': adversarial_confidence,
            'image_data': None  # Will be filled later
        }
        
        return adv_image, attack_data
    
    def visualize_results(self, original_image: torch.Tensor, 
                         adversarial_image: torch.Tensor, 
                         attack_data: Dict):
        """Visualize attack results."""
        # Convert tensors to numpy arrays
        orig_img = original_image.squeeze().cpu().numpy()
        adv_img = adversarial_image.squeeze().cpu().numpy()
        
        # Calculate perturbation
        perturbation = adv_img - orig_img
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Original Image', 'Adversarial Image', 'Perturbation'],
            horizontal_spacing=0.1
        )
        
        # Original image
        fig.add_trace(
            go.Heatmap(z=orig_img, colorscale='gray', showscale=False),
            row=1, col=1
        )
        
        # Adversarial image
        fig.add_trace(
            go.Heatmap(z=adv_img, colorscale='gray', showscale=False),
            row=1, col=2
        )
        
        # Perturbation
        fig.add_trace(
            go.Heatmap(z=perturbation, colorscale='RdBu', 
                      zmid=0, showscale=True),
            row=1, col=3
        )
        
        fig.update_layout(
            height=300,
            showlegend=False,
            title_text=f"Attack Results - {attack_data['method'].upper()}",
            title_x=0.5
        )
        
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Original Prediction",
                attack_data['original_pred'],
                f"Confidence: {attack_data['confidence_original']:.3f}"
            )
        
        with col2:
            st.metric(
                "Adversarial Prediction",
                attack_data['adversarial_pred'],
                f"Confidence: {attack_data['confidence_adversarial']:.3f}"
            )
        
        with col3:
            success_text = "✅ Success" if attack_data['success'] else "❌ Failed"
            st.metric("Attack Success", success_text)
        
        with col4:
            st.metric("True Label", attack_data['true_label'])
    
    def render_attack_generator(self):
        """Render the main attack generator interface."""
        st.header("🎯 Generate Adversarial Examples")
        
        # Get configuration from sidebar
        config = self.render_sidebar()
        
        # Image selection
        st.subheader("📸 Select Input Image")
        
        # Option 1: Upload custom image
        uploaded_file = st.file_uploader(
            "Upload MNIST-style image (28x28 grayscale)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a 28x28 grayscale image"
        )
        
        # Option 2: Use random MNIST sample
        use_random = st.checkbox("Use random MNIST sample", value=True)
        
        if uploaded_file is not None:
            # Process uploaded image
            image = Image.open(uploaded_file).convert('L').resize((28, 28))
            image_array = np.array(image) / 255.0
            image_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)
            
            # For uploaded images, we don't know the true label
            true_label = st.number_input("True Label (0-9)", min_value=0, max_value=9, value=0)
            label_tensor = torch.LongTensor([true_label])
            
        elif use_random:
            # Load random MNIST sample
            from torchvision import datasets, transforms
            
            transform = transforms.ToTensor()
            test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            
            # Get random sample
            if 'random_idx' not in st.session_state:
                st.session_state.random_idx = np.random.randint(0, len(test_dataset))
            
            if st.button("🔄 New Random Sample"):
                st.session_state.random_idx = np.random.randint(0, len(test_dataset))
            
            sample, label = test_dataset[st.session_state.random_idx]
            image_tensor = sample.unsqueeze(0)
            label_tensor = torch.LongTensor([label])
            
            st.write(f"Random MNIST sample #{st.session_state.random_idx}")
        
        else:
            st.warning("Please upload an image or select 'Use random MNIST sample'")
            return
        
        # Display original image
        if uploaded_file is not None or use_random:
            st.subheader("📊 Original Image")
            orig_img = image_tensor.squeeze().cpu().numpy()
            
            fig_orig = go.Figure(data=go.Heatmap(z=orig_img, colorscale='gray', showscale=False))
            fig_orig.update_layout(
                title="Original Image",
                height=300,
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False)
            )
            st.plotly_chart(fig_orig, use_container_width=True)
        
        # Generate adversarial example
        if st.button("🚀 Generate Adversarial Example", type="primary"):
            if not st.session_state.model_loaded:
                st.error("Please train a model first!")
                return
            
            with st.spinner("Generating adversarial example..."):
                try:
                    adversarial_image, attack_data = self.generate_adversarial_example(
                        image_tensor, label_tensor, config
                    )
                    
                    # Save result to database
                    self.save_attack_result(attack_data)
                    
                    # Visualize results
                    st.subheader("🎯 Attack Results")
                    self.visualize_results(image_tensor, adversarial_image, attack_data)
                    
                    # Display attack details
                    st.subheader("📋 Attack Details")
                    attack_df = pd.DataFrame([attack_data])
                    st.dataframe(attack_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error generating adversarial example: {e}")
    
    def render_statistics(self):
        """Render attack statistics dashboard."""
        st.header("📊 Attack Statistics Dashboard")
        
        # Get statistics from database
        stats = self.get_attack_statistics()
        
        if stats['total_attacks'] == 0:
            st.info("No attacks have been performed yet. Generate some adversarial examples first!")
            return
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Attacks", stats['total_attacks'])
        
        with col2:
            success_rate = stats['successful_attacks'] / stats['total_attacks']
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        with col3:
            st.metric("Average Epsilon", f"{stats['avg_epsilon']:.3f}")
        
        with col4:
            st.metric("Attack Methods Used", len(stats['method_stats']))
        
        # Method-wise statistics
        st.subheader("📈 Attack Method Performance")
        
        method_df = pd.DataFrame([
            {'Method': method, 'Count': count} 
            for method, count in stats['method_stats'].items()
        ])
        
        fig_methods = px.bar(method_df, x='Method', y='Count', 
                           title="Attacks by Method")
        st.plotly_chart(fig_methods, use_container_width=True)
        
        # Recent attacks table
        st.subheader("🕒 Recent Attacks")
        
        conn = sqlite3.connect(self.db_path)
        recent_attacks = pd.read_sql_query("""
        SELECT timestamp, attack_method, epsilon, original_prediction, 
               adversarial_prediction, true_label, success
        FROM attack_results 
        ORDER BY timestamp DESC 
        LIMIT 10
        """, conn)
        conn.close()
        
        if not recent_attacks.empty:
            st.dataframe(recent_attacks, use_container_width=True)
        else:
            st.info("No recent attacks found.")
    
    def render_model_training(self):
        """Render model training interface."""
        st.header("🏋️ Model Training")
        
        if not st.session_state.model_loaded:
            st.warning("No pre-trained model found. Train a new model to get started!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Training Configuration")
            
            model_type = st.selectbox("Model Type", ["modern", "simple"])
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001)
            batch_size = st.selectbox("Batch Size", [32, 64, 128], index=1)
            num_epochs = st.slider("Number of Epochs", 1, 20, 5)
        
        with col2:
            st.subheader("📊 Training Status")
            
            if st.session_state.model_loaded:
                st.success("✅ Pre-trained model loaded")
            else:
                st.error("❌ No pre-trained model")
            
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        # Initialize trainer
                        trainer = Trainer(
                            model_type=model_type,
                            device='cpu',
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            num_epochs=num_epochs,
                            save_dir='./checkpoints'
                        )
                        
                        # Train model
                        history = trainer.train()
                        
                        # Update session state
                        st.session_state.model = trainer.model
                        st.session_state.model_loaded = True
                        
                        st.success("✅ Model training completed!")
                        
                        # Display training results
                        st.subheader("📈 Training Results")
                        
                        final_acc = history['val_accuracies'][-1]
                        best_acc = history['best_val_acc']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Final Accuracy", f"{final_acc:.2f}%")
                        with col2:
                            st.metric("Best Accuracy", f"{best_acc:.2f}%")
                        
                        # Plot training history
                        fig = make_subplots(rows=1, cols=2, subplot_titles=['Loss', 'Accuracy'])
                        
                        epochs = list(range(1, len(history['train_losses']) + 1))
                        
                        fig.add_trace(
                            go.Scatter(x=epochs, y=history['train_losses'], 
                                     name='Train Loss', line=dict(color='blue')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=epochs, y=history['val_losses'], 
                                     name='Val Loss', line=dict(color='red')),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=epochs, y=history['train_accuracies'], 
                                     name='Train Acc', line=dict(color='blue')),
                            row=1, col=2
                        )
                        fig.add_trace(
                            go.Scatter(x=epochs, y=history['val_accuracies'], 
                                     name='Val Acc', line=dict(color='red')),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=400, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during training: {e}")
    
    def run(self):
        """Run the main application."""
        self.render_header()
        
        # Navigation
        tab1, tab2, tab3 = st.tabs(["🎯 Attack Generator", "📊 Statistics", "🏋️ Model Training"])
        
        with tab1:
            self.render_attack_generator()
        
        with tab2:
            self.render_statistics()
        
        with tab3:
            self.render_model_training()


def main():
    """Main function to run the Streamlit app."""
    app = AdversarialAttackUI()
    app.run()


if __name__ == "__main__":
    main()
