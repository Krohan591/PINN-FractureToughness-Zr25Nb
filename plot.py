import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def plot_training_and_predictions(model, X_test, y_test, p_loss, d_loss, t_loss, best_weight):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test.numpy(), y_pred)
    mae = mean_absolute_error(y_test.numpy(), y_pred)
    mse = mean_squared_error(y_test.numpy(), y_pred)

    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # --- Plot 1: Loss Curves ---
    axs[0].plot(p_loss, label='Physics Loss')
    axs[0].plot(d_loss, label='Data Loss')
    axs[0].plot(t_loss, label='Total Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title(f'Loss Curves for PINN (λ = {best_weight})')
    axs[0].legend()

    # --- Plot 2: True vs Predicted ---
    axs[1].plot(y_test.numpy(), y_pred, 'o', alpha=0.6, label='PINN', markersize=10,
                color='orange', markeredgecolor='darkorange')
    axs[1].plot([0, 1], [0, 1], '--', color='gray')
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 1])
    axs[1].set_xlabel('True Values')
    axs[1].set_ylabel('Predicted Values')
    axs[1].set_title('True vs Predicted')
    axs[1].legend()
    axs[1].text(0.05, 0.85, f"R² Score: {r2:.3f}", transform=axs[1].transAxes,
                fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"))

    plt.tight_layout()
    plt.show()
