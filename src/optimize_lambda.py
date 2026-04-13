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

def optimize_lambda(train_loader, val_loader, FROGNet, n_trials=20):
    print("מתחיל שלב 1: כוונון למדא בעזרת Optuna על הארכיטקטורה המהירה...")

    def objective(trial):
        # 1. הגדרת טווח החיפוש עבור למדא (למשל 0.0 עד 1.0)
        lambda_val = trial.suggest_float('lambda', 0.0, 1.0)

        # שימוש בארכיטקטורה המהירה (Multires) עבור הכוונון
        model = Multires().cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # יוצר אובייקט פונקציית הפסד מסוג L1 (Mean Absolute Error), עם reduction='sum' לסיכום ההפסדים לשגיאה כוללת אחת של האצווה.
        l1_loss = nn.L1Loss(reduction='sum')

        # אימון קצר (למשל 5 אפוקים) להערכת הלמדא
        model.train()
        for epoch in range(5):
            for I_clean, E_label in train_loader:
                I_clean, E_label = I_clean.cuda(), E_label.cuda()

                # הזרקת רעש אקראי מ-0 עד 30 dB (בהתאם למאמר)
                snr = np.random.uniform(0, 30)
                I_noisy = add_noise(I_clean, snr, "WGN")

                optimizer.zero_grad()
                E_pred = model(I_noisy)

                # חישוב השגיאה המשולבת
                loss_supervised = l1_loss(E_pred, E_label)
                I_reconstructed = FROGNet(E_pred)
                loss_unsupervised = l1_loss(I_reconstructed, I_noisy)

                total_loss = loss_supervised + lambda_val * loss_unsupervised
                # מבצע backpropagation: מחשב את הגרדיאנטים של ההפסד הכולל ביחס למשקולות המודל.
                total_loss.backward()
                # מעדכן את משקולות המודל באמצעות הגרדיאנטים המחושבים ושיטת האופטימיזציה (Adam).
                optimizer.step()

        # חישוב שגיאת ולידציה
        # מעביר את המודל למצב הערכה (מכבה Dropout, Batch Norm וכדומה).
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for I_clean, E_label in val_loader:
                I_clean, E_label = I_clean.cuda(), E_label.cuda()
                E_pred = model(I_clean)
                val_loss += l1_loss(E_pred, E_label).item()

        # מחזיר את ממוצע שגיאת הולידציה למנה (batch).
        return val_loss / len(val_loader)

    # יוצר אובייקט מחקר חדש ב-Optuna, שמטרתו למזער (minimize) את פונקציית המטרה.
    study = optuna.create_study(direction='minimize')
    # מריץ את תהליך האופטימיזציה של Optuna על פונקציית המטרה למספר ניסיונות (trials) שצוין.
    study.optimize(objective, n_trials=n_trials)

    # שולף את הערך של הלמדא שהניב את התוצאה הטובה ביותר (ההפסד הממוזער ביותר) מבין כל הניסיונות.
    lambda_opt = study.best_params['lambda']
    print(f"הערך האופטימלי שנמצא: lambda = {lambda_opt}")

    # יוצר רשימה של מספרי הניסיונות שבוצעו על ידי Optuna.
    trials = [t.number for t in study.trials]
    # יוצר רשימה של ערכי פונקציית המטרה (הפסדי ולידציה) שהתקבלו עבור כל ניסיון.
    values = [t.value for t in study.trials]
    plt.plot(trials, values, marker='o')
    plt.title("Optimization History (Validation Loss vs. Trial)")
    plt.xlabel("Trial")
    plt.ylabel("Validation Loss")
    plt.show()

    return lambda_opt