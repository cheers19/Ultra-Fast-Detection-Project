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

# הקוד המעודכן עם פרמטרים פתוחים לארכיטקטורה וטווח SNR
def optimize_lambda(train_loader, val_loader, FROGNet, model_architecture, snr_range=(0, 30), n_trials=20):
    print("מתחיל שלב 1: כוונון למדא בעזרת Optuna...")
    
    def objective(trial):
        lambda_val = trial.suggest_float('lambda', 0.0, 1.0)
        
        # אתחול הארכיטקטורה שהמשתמש העביר כפרמטר
        model = model_architecture().cuda() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        l1_loss = nn.L1Loss(reduction='sum')
        
        model.train()
        for epoch in range(5):
            for I_clean, E_label in train_loader:
                I_clean, E_label = I_clean.cuda(), E_label.cuda()
                
                # הזרקת רעש אקראי בטווח שהמשתמש העביר
                snr = np.random.uniform(snr_range, snr_range[1])
                I_noisy = add_noise(I_clean, snr, "WGN")
                
                optimizer.zero_grad()
                E_pred = model(I_noisy)
                
                loss_supervised = l1_loss(E_pred, E_label)
                I_reconstructed = FROGNet(E_pred)
                loss_unsupervised = l1_loss(I_reconstructed, I_noisy)
                
                total_loss = loss_supervised + lambda_val * loss_unsupervised
                total_loss.backward()
                optimizer.step()
                
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for I_clean, E_label in val_loader:
                I_clean, E_label = I_clean.cuda(), E_label.cuda()
                E_pred = model(I_clean)
                val_loss += l1_loss(E_pred, E_label).item()
                
        return val_loss / len(val_loader)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    lambda_opt = study.best_params['lambda']
    print(f"הערך האופטימלי שנמצא: lambda = {lambda_opt}")
    
    # עדכון קל: הגרף כעת יציג שגיאה כפונקציה של למדא (ראה תשובה לשאלה 16)
    lambdas = [t.params['lambda'] for t in study.trials]
    values = [t.value for t in study.trials]
    plt.scatter(lambdas, values)
    plt.title("Validation Loss vs. Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Validation Loss")
    plt.show()
    
    return lambda_opt