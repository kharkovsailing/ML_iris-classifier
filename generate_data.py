import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load built-in iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Split into training and inference sets
df_train, df_infer = train_test_split(df, test_size=0.3, random_state=42, stratify=df["species"])

# Save to CSV
df_train.to_csv("data/iris_train.csv", index=False)
df_infer.drop(columns=["species"]).to_csv("data/iris_inference.csv", index=False)

print(" Data files saved: iris_train.csv and iris_inference.csv")
