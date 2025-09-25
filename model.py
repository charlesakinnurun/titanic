# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

# %% [markdown]
# Set a style for the plots for better visualization

# %%
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("viridis")

# %% [markdown]
# Data Loading and Exploration

# %%
try:
    df = pd.read_csv("data.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: The 'data.csv' was not found. Check the if it is present in the current directory")
    exit()


# Display the first few rows of the dataframe to understand its structure
print("Initial Data Preview")
df.head()

# %%
# Get the summary of the dataframe to check for missing values and data types
print("Data Information")
print(df.info())

# %% [markdown]
# Visualization before Training

# %%
# Create a figure with subplots 
fig,axes = plt.subplots(1,3,figsize=(18,6))

# Plot 1 : Survival rate by Sex
sns.barplot(x="Sex",y="Survived",data=df,ax=axes[0])
axes[0].set_title("Survial Rate by Gender")
axes[0].set_xlabel("Sex")
axes[0].set_ylabel("Survival Rate")

# Plot 2: Survival rate by Passenger Class (Pclass)
sns.barplot(x="Pclass",y="Survived",data=df,ax=axes[1])
axes[1].set_title("Survial Rate by Passenger Class")
axes[1].set_xlabel("Passenger Class")
axes[0].set_ylabel("Survival Rate")

# Plot 3: Survival rate by Age distriution (using a histogram)
sns.histplot(x="Age",hue="Survived",data=df,multiple="stack",ax=axes[2])
axes[2].set_title("Survival Rate by Age")
axes[2].set_xlabel("Age")
axes[2].set_ylabel("Count")

# Adjust layout to prevent titles from overlapping
plt.tight_layout()

# Show the plots 
plt.show()

# %% [markdown]
# Data Preprocessing and Feature Engineering

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# Change the missing values in the "Age" column to 0
df["Age"] = df["Age"].fillna(0)

# Change the missing values in the "Cabin" column to "-"
df["Cabin"] = df["Cabin"].fillna("-")

# Change the missing values in the "Embarked" column to "-"
df["Embarked"] = df["Embarked"].fillna("-")

# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated Rows")
print(df_duplicated)

# Define the features (X) and the target variable (y)
features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
target = "Survived"

X = df[features]
y = df[target]

# Define the features are numerical and which are categorical
numerical_features = ["Age","Fare","SibSp","Parch"]
categorical_features = ["Pclass","Sex","Embarked"]

# Create a Preprocessing pipeline
# The pipeline will handle missing values, scaling anfd one-hot encoding
numerical_transformer = Pipeline(steps=[
    # Impute missing numerical values with the mean
    ("imputer",SimpleImputer(strategy="mean")),
    # Standardize features by removing the mean and scaling to unit variance
    ("scaler",StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    # Impute missing categorical values with the most frequent value
    ("imputer",SimpleImputer(strategy="most_frequent")),
    # One-hot encode categorical features to numerical format
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num",numerical_transformer,numerical_features),
        ("cat",categorical_transformer,categorical_features)
    ]
)

# %% [markdown]
# Training and Evaluating Models

# %%
print("Training and evaluating various clssification models.....")

# Dictionary to store the models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "K-nearest Neighbours": KNeighborsClassifier(),
    "Support Vector Machine": SVC(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Dictionary to store the models
performance_metrics = {
    "Accuracy": [],
    "Precison": [],
    "Recall": [],
    "F1 Score": []
}

# Split the data into training and testing sets (80% training, 20% testing)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Loop through each model to train and evaluate
trained_models = {}
for name,model in models.items():
    print(f" Training {name}.....")

    # Create a full pipeline that first preprocesses, then trains the model
    full_pipeline = Pipeline(steps=[("preprocessor",preprocessor),
                                    ("classifier",model)])
    
    # Fit the pipeline to the training data
    full_pipeline.fit(X_train,y_train)
    trained_models[name] = full_pipeline

    # Make predictions on the test set
    y_pred = full_pipeline.predict(X_test)

    # Calculate and store the performance metrics
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    performance_metrics["Accuracy"].append(accuracy)
    performance_metrics["Precison"].append(precision)
    performance_metrics["Recall"].append(recall)
    performance_metrics["F1 Score"].append(f1)


    print(f"     - {name} Metrics")
    print(f"     Accuracy: {accuracy:.4f}")
    print(f"     Precison: {precision:.4f}")
    print(f"     Recall: {recall:.4f}")
    print(f"     F1 Score: {f1:.4f}")
    print("-"* 30)


# Create a DataFrame to display the performance metrics clearly
metrics_df = pd.DataFrame(performance_metrics,index=models.keys())
print("Model Performance Summary:")
print(metrics_df.sort_values(by="Accuracy",ascending=False))

# %% [markdown]
# Visulaization of Model Performance (After Training)

# %%
# Create a figure with subplots for each metric
fig,axes = plt.subplots(2,2,figsize=(16,12))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through each metric and plot a bar chart
for i, metric in enumerate(performance_metrics.keys()):
    sns.barplot(x=metrics_df.index,y=metrics_df[metric],ax=axes[i])
    axes[i].set_title(f"Comparison of {metric}", fontsize=14)
    axes[i].set_ylabel(metric)
    axes[i].set_xlabel("Model")
    # Rotate x-axis labels for readablity
    axes[i].tick_params(axis="x",rotation=45)

# Adjust layut to prevent titles from overlapping
plt.tight_layout()

# Show the plots
plt.show()

# %% [markdown]
# Find the best Model

# %%
# Find the best model based on F1-Score (a good balance of precison and recall)
best_model_name = metrics_df["F1 Score"].idxmax()
best_model = trained_models[best_model_name]
print(f"The best performing model based on F1 Score is {best_model_name}")

# %% [markdown]
# Intercative User Prediction

# %%
print("Now, let's make a prediction for a new passenger using the best model")
print("Please provide the following information:")

try:
    # Get user input for each feature
    pclass_input = int(input("Enter Passenger Class (1, 2, or 3):"))
    sex_input = (input("Enter Sex (male or female):"))
    age_input = float(input("Enter Age: "))
    sibsp_input = int(input("Enter number of Siblings/Spouses Aboard:"))
    parch_input = int(input("Enter number of Parents/Children Aboard:"))
    fare_input = float(input("Enter Fare: "))
    embarked_input = input("Enter Port of Emabarkation (C, Q, or S):")

    # Create a DataFrame with the user input
    user_data = pd.DataFrame([[pclass_input, sex_input, age_input, sibsp_input,
                               parch_input, fare_input, embarked_input]],
                             columns=features)
    # Use the best model to make a prediction
    prediction = best_model.predict(user_data)

    # Interpret the prediction result
    if prediction[0] == 1:
        print("Prediction: The Passenger would likely have SURVIVED.")
    else:
        print("Prediction: The passenger would likely have NOT SURVIVED.")

    # Show the confidence of the prediction if possible
    if hasattr(best_model, "predict_proba"):
        prediction_proba = best_model.predict_proba(user_data)
        print(f"Survival Probability: {prediction_proba[0][1]:.2f}")
        print(f"Non-Survival Probability: {prediction_proba[0][0]:.2f}")
    else:
        print("Probability prediction is not available for this model.")

except ValueError:
    print("Invalid input. Please ensure you enter the correct data types.")


