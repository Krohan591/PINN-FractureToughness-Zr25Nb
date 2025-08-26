import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def plot_error_bands():
    x = np.linspace(0, 1, 10)
    y = x

    # Plot 30% error band lines
    plt.plot(x, y + 0.2, '--', label='±20% Error Band',color='yellowgreen',linewidth=1.5)
    plt.plot(x, y - 0.2, '--',color='yellowgreen',linewidth=1.5)

    # Plot 20% error band lines
    plt.plot(x, y + 0.1, '-',color='dodgerblue', label='±10% Error Band',linewidth=0.8)
    plt.plot(x, y - 0.1, '-',color='dodgerblue',linewidth=0.8)

    # Plot ideal line

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pinn_performance(y_test, y_pred, p_loss, d_loss, t_loss, best_weight):
    if hasattr(y_test, 'numpy'):
        y_test = y_test.numpy()
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

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
    axs[1].plot(y_test, y_pred, 'o', alpha=0.6, label='PINN', markersize=10,
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
    plot_error_bands()
    plt.tight_layout()
    plt.show()

def plot_ann_performance(y_test, y_pred, history, model_name='ANN'):

    # Convert tensors to numpy if needed
    if hasattr(y_test, 'numpy'):
        y_test = y_test.numpy()

    # Evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Plot setup
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- Subplot 1: Loss Curves ---
    axs[0].plot(history.history['loss'], label='Train Loss')
    axs[0].plot(history.history['val_loss'], label='Val Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Mean Squared Error')
    axs[0].set_title('Training vs Validation Loss')
    axs[0].legend()

    # --- Subplot 2: True vs Predicted ---
    axs[1].plot(y_test, y_pred, 'o', alpha=0.4, markersize=10, color='purple', label=model_name)
    axs[1].plot([0, 1], [0, 1], '--', color='black')  # Reference line
    axs[1].set_xlabel('True Values')
    axs[1].set_ylabel('Predicted Values')
    axs[1].set_title('True vs Predicted')

    # Display metrics inside the plot
    metrics_text = f"R² Score: {r2:.3f}\nMAE: {mae:.3f}\nMSE: {mse:.3f}"
    axs[1].text(0.05, 0.85, metrics_text, transform=axs[1].transAxes,
                fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"))
    
    axs[1].legend()
    plt.tight_layout()
    plot_error_bands()
    # # Optional: Plot error bands if function is passed
    # if callable(plot_error_bands):
    #     plot_error_bands(ax=axs[1])

    plt.show()

def plot_pred_vs_true(y_true, y_pred, label='Model', color='tomato', edgecolor='red' ,show_error_band=True):

    # Convert to NumPy if using tensors
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()

    r2 = r2_score(y_true, y_pred)

    plt.style.use('ggplot')
    # plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], '--', color='black', alpha=0.8)
    plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.7, label=label,
                edgecolors=edgecolor, color=color, s=100)
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.text(0.05, 0.85, f"R² Score: {r2:.3f}", transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"))
    
    plt.legend()
    plt.tight_layout()

    if show_error_band and 'plot_error_bands' in globals():
        plot_error_bands()

    plt.show()
