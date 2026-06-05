import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt

from utils import add_noise

# ==========================================
# Step 1: hyperparameter tuning (optimal lambda search)
# ==========================================

# Updated code: Switch to reduction='mean' for supervised/unsupervised L1 losses (Equation 2) to normalize by batch size. Early Stopping remains enabled.
def optimize_lambda(train_loader, val_loader, FROGNet, model_architecture, snr_range=(0, 30), n_trials=20, max_epochs=50, patience=5):
    print("Starting step 1: tuning lambda with Optuna (mean L1 error and early stopping)...")
    
    def objective(trial):
        lambda_val = trial.suggest_float('lambda', 0.0, 1.0)
        
        # Initialize the architecture
        model = model_architecture().cuda() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Use mean reduction instead of sum
        l1_loss = nn.L1Loss(reduction='mean')
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # Train with a high max epoch count (for early stopping)
        for epoch in range(max_epochs):
            model.train()
            for I_clean, E_label in train_loader:
                I_clean, E_label = I_clean.cuda(), E_label.cuda()
                
                # Inject noise for training
                snr = np.random.uniform(snr_range, snr_range[2])
                I_noisy = add_noise(I_clean, snr, "WGN")
                
                optimizer.zero_grad()
                E_pred = model(I_noisy)
                
                # Combined losses per Equation 2, now computed as mean [1]
                loss_supervised = l1_loss(E_pred, E_label)
                I_reconstructed = FROGNet(E_pred)
                loss_unsupervised = l1_loss(I_reconstructed, I_noisy)
                
                total_loss = loss_supervised + lambda_val * loss_unsupervised
                total_loss.backward()
                optimizer.step()
                
            # Validation check at the end of each epoch
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for I_clean, E_label in val_loader:
                    I_clean, E_label = I_clean.cuda(), E_label.cuda()
                    E_pred = model(I_clean)
                    val_loss += l1_loss(E_pred, E_label).item()
            
            val_loss /= len(val_loader)
            
            # --- Early stopping logic ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0  # reset counter on improvement
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    # exit loop if no improvement for 'patience' consecutive epochs
                    break 
                
        # Return the best validation loss achieved
        return best_val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    lambda_opt = study.best_params['lambda']
    print(f"Optimal value found: lambda = {lambda_opt}")
    
    # Plot results
    lambdas = [t.params['lambda'] for t in study.trials]
    values = [t.value for t in study.trials]
    plt.scatter(lambdas, values)
    plt.title("Validation Loss vs. Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Validation Loss")
    plt.show()
    
    return lambda_opt
