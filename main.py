import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

# Load dataset
path = kagglehub.dataset_download("kartiksaini18/churn-bank-customer")
df = pd.read_csv(path + "/Churn_Modelling.csv")

# Features engineering
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

X = df.drop("Exited", axis=1)
y = df["Exited"]
feature_names = X.columns

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# === Logistic Regression ===
print("Logistic Regression")
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
print("ROC AUC:", roc_auc_score(y_test, lr_proba))

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
plt.figure(figsize=(8, 5))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, lr_proba):.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve — Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# === Gradient Boosting ===
print("Gradient Boosting")
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_proba = gb.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, gb_pred))
print(classification_report(y_test, gb_pred))
print("ROC AUC:", roc_auc_score(y_test, gb_proba))

fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_proba)
plt.figure(figsize=(8, 5))
plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC = {roc_auc_score(y_test, gb_proba):.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve — Gradient Boosting")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Feature Importance — Gradient Boosting
gb_importances = gb.feature_importances_
indices_gb = np.argsort(gb_importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importance — Gradient Boosting")
plt.bar(range(len(gb_importances)), gb_importances[indices_gb], align="center", color="green")
plt.xticks(range(len(gb_importances)), feature_names[indices_gb], rotation=90)
plt.tight_layout()
plt.grid(True)
plt.show()


#
churned_age_stats = df[df["Exited"] == 1]["Age"].describe()
print("Avg age of Churns:")
print(churned_age_stats)





df_churned = df[df["Exited"] == 1]

plt.figure(figsize=(6, 3))  
sns.boxplot(data=df_churned, x="Age", color="coral", width=0.4)

plt.title("Boxplot: Age of Churned Customers", fontsize=12)
plt.xlabel("Age")
plt.yticks([])  
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))

# KDE plot 
sns.kdeplot(data=df[df["Exited"] == 1], x="NumOfProducts", label="Churned", fill=True, color="tomato")
sns.kdeplot(data=df[df["Exited"] == 0], x="NumOfProducts", label="Stayed", fill=True, color="skyblue")

plt.title("KDE: Distribution of Number of Products by Churn Status")
plt.xlabel("Number of Products")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


churned_products_stats = df[df["Exited"] == 1]["NumOfProducts"].describe()
print("Avg NumOfProducts of Churned Customers:")
print(churned_products_stats)

stayed_products_stats = df[df["Exited"] == 0]["NumOfProducts"].describe()
print("\nAvg NumOfProducts of Stayed Customers:")
print(stayed_products_stats)
