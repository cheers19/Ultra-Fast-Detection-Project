import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import matplotlib.pyplot as plt

from utils import add_noise

# ==========================================
# שלב 1: כוונון היפר-פרמטרים (חיפוש למדא אופטימלי)
# ==========================================

# Updated code with open parameters for architecture, SNR range, and early stopping
def optimize_lambda(train_loader, val_loader, FROGNet, model_architecture, snr_range=(0, 30), n_trials=20, max_epochs=50, patience=5):
    print("מתחיל שלב 1: כוונון למדא בעזרת Optuna עם עצירה מוקדמת...")
    
    def objective(trial):
        lambda_val = trial.suggest_float('lambda', 0.0, 1.0)
        
        # אתחול הארכיטקטורה
        model = model_architecture().cuda() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        l1_loss = nn.L1Loss(reduction='sum')
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # אימון עם מספר אפוקים מקסימלי גבוה
        for epoch in range(max_epochs):
            model.train()
            for I_clean, E_label in train_loader:
                I_clean, E_label = I_clean.cuda(), E_label.cuda()
                
                # הזרקת רעש לאימון
                snr = np.random.uniform(snr_range, snr_range[2])
                I_noisy = add_noise(I_clean, snr, "WGN")
                
                optimizer.zero_grad()
                E_pred = model(I_noisy)
                
                loss_supervised = l1_loss(E_pred, E_label)
                I_reconstructed = FROGNet(E_pred)
                loss_unsupervised = l1_loss(I_reconstructed, I_noisy)
                
                total_loss = loss_supervised + lambda_val * loss_unsupervised
                total_loss.backward()
                optimizer.step()
                
            # בדיקת אימות בסוף כל אפוק
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for I_clean, E_label in val_loader:
                    I_clean, E_label = I_clean.cuda(), E_label.cuda()
                    E_pred = model(I_clean)
                    val_loss += l1_loss(E_pred, E_label).item()
            
            val_loss /= len(val_loader)
            
            # --- לוגיקת עצירה מוקדמת (Early Stopping) ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0  # איפוס המונה אם יש שיפור
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    # יציאה מהלולאה אם אין שיפור למשך 'patience' אפוקים רצופים
                    break 
                
        # מחזירים את התוצאה הטובה ביותר שהושגה
        return best_val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    lambda_opt = study.best_params['lambda']
    print(f"הערך האופטימלי שנמצא: lambda = {lambda_opt}")
    
    # הפקת הגרף
    lambdas = [t.params['lambda'] for t in study.trials]
    values = [t.value for t in study.trials]
    plt.scatter(lambdas, values)
    plt.title("Validation Loss vs. Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Validation Loss")
    plt.show()
    
    return lambda_opt