#Support Vector Machines (SVM) for Binary Classification
#Dataset: Wisconsin Breast Cancer Dataset (data.csv)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# 1. Load dataset (569 samples, 30 features)​

df = pd.read_csv("data.csv")

# Drop the 'id' column (not a feature)
df.drop(columns=["id"], inplace=True)

# Check for any unnamed/empty trailing columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print(f"\nDataset Shape: {df.shape}")
print(f"Features: {df.shape[1] - 1}")
print(f"Samples : {df.shape[0]}")
print(f"\nTarget Distribution:")
print(df['diagnosis'].value_counts())
print(f"\nMissing Values: {df.isnull().sum().sum()}")

# 2. DATA PREPROCESSING

le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
print(f"\nLabel Encoding: B → 0, M → 1")

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set : {X_test.shape[0]} samples")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. SVM MODEL TRAINING WITH DIFFERENT KERNELS

kernels = {
    'Linear': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
    'RBF': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    'Polynomial': SVC(kernel='poly', degree=3, C=1.0, probability=True, random_state=42),
    'Sigmoid': SVC(kernel='sigmoid', C=1.0, probability=True, random_state=42),
}

results = {}

print("\n" + "=" * 60)
print("  MODEL TRAINING & EVALUATION")
print("=" * 60)

for name, model in kernels.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'accuracy': acc
    }
    print(f"\n{'─' * 50}")
    print(f"  Kernel: {name} | Accuracy: {acc:.4f}")
    print(f"{'─' * 50}")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# 4. MODEL SUMMARY?


best_kernel = max(results, key=lambda k: results[k]['accuracy'])
best_acc = results[best_kernel]['accuracy']

print("\n" + "=" * 60)
print(f"  BEST MODEL: {best_kernel} Kernel (Accuracy: {best_acc:.4f})")
print("=" * 60)


# 5. GRAPHS


fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('SVM Classification Results — Confusion Matrices', fontsize=16, fontweight='bold')

for idx, (name, res) in enumerate(results.items()):
    ax = axes[idx // 2, idx % 2]
    cm = confusion_matrix(y_test, res['y_pred'])
    ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant']).plot(
        ax=ax, cmap='Blues', colorbar=False
    )
    ax.set_title(f"{name} Kernel (Acc: {res['accuracy']:.4f})", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('svm_confusion_matrices.png', dpi=150, bbox_inches='tight')
print("\nSaved: outputs/svm_confusion_matrices.png")

# --- ROC Curves ---
plt.figure(figsize=(10, 8))
for name, res in results.items():
    y_prob = res['model'].predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves — SVM Kernels Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('svm_roc_curves.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/svm_roc_curves.png")

# --- Accuracy Comparison Bar Chart ---
plt.figure(figsize=(8, 5))
names = list(results.keys())
accs = [results[k]['accuracy'] for k in names]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
bars = plt.bar(names, accs, color=colors, edgecolor='black', linewidth=0.5)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.ylim(min(accs) - 0.1, 1.02)
plt.ylabel('Accuracy', fontsize=12)
plt.title('SVM Kernel Accuracy Comparison', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('svm_accuracy_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/svm_accuracy_comparison.png")

# --- Feature Importance (Linear Kernel Coefficients) ---
linear_model = results['Linear']['model']
feature_importance = np.abs(linear_model.coef_[0])
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)[-15:]  # Top 15

plt.figure(figsize=(10, 7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='#3F51B5')
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx], fontsize=10)
plt.xlabel('Absolute Coefficient Value', fontsize=12)
plt.title('Top 15 Feature Importances (Linear SVM)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('svm_feature_importance.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/svm_feature_importance.png")

print("ALL GRAPHS PRESENT IN /OUTPUTS")
