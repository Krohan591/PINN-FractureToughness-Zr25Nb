# PINN-Based Fracture Toughness Prediction

This repository provides a Physics-Informed Neural Network (PINN) framework to predict the fracture toughness $\(K_{IC}\)$ of Zr-2.5Nb alloys under varying hydrogen exposure conditions. The model integrates physical domain knowledge and experimental data, achieving enhanced generalization and physical consistency even in data-sparse regimes.

## 📁 Project Structure

```
├── main.py                # Entry point: trains, saves, loads, and evaluates the PINN
├── models.py              # PINN architecture (custom Keras model with physics constraints)
├── losses.py              # Custom physics loss using gradients and empirical sigmoid behavior
├── tune.py                # Hyperparameter tuning using Optuna
├── plot_models.py         # Visualization of training loss and model predictions
├── Example_case.ipynb     # Comparing the performance of PINNs with other machine learning approaches.
└── README.md              # Project overview and setup instructions
```

## ⚙️ Features

- PINN implementation using TensorFlow 
- Physics loss based on sigmoid transition of fracture behavior with temperature
- Hyperparameter tuning using Optuna
- True vs. predicted plotting
- Model saving embedded within `main.py`

## 🧪 Dependencies

Install dependencies,Required packages include:
- `tensorflow`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `optuna`

## 🚀 Quick Start

```bash
python main.py
```

This will:
- Train the PINN
- Save it to `pinn_model.keras`
- Plot the loss curves and predicted vs. actual fracture toughness

## 📈 Results
- Achieves robust predictions with limited data
- Maintains physical consistency in predictions (e.g., reduction in toughness with increased hydrogen)

## 🔬 References
This work builds on the idea of embedding empirical physics into deep learning, particularly via PINNs.

---

📧 For any questions or suggestions, feel free to open an issue or contact the maintainer.


