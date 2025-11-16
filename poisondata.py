import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import argparse

# --------------------
# Take user input
# --------------------
parser = argparse.ArgumentParser(description="Poison Iris dataset labels")
parser.add_argument(
    "--poison_fraction",
    type=float,
    required=True,
    help="Fraction of labels to poison (0–1). Example: 0.1 = 10%"
)

args = parser.parse_args()
poison_fraction = args.poison_fraction




# Load iris dataset
df = pd.read_csv("data/data_clean.csv")

# Poisoning parameters
#poison_fraction = 0.05  # 5% poisoning
num_rows = len(df)
num_poison = int(num_rows * poison_fraction)  # 150 samples → 7 poisoned samples

# Randomly choose rows to poison
np.random.seed(42)
poison_indices = np.random.choice(num_rows, num_poison, replace=False)

# Create poisoned dataset copy
df_poisoned = df.copy()

# Label range
label_values = np.unique(df["species"])

# Inject incorrect labels
for idx in poison_indices:
    true_label = df.loc[idx, "species"]
    
    # Choose a random incorrect label
    incorrect_choices = label_values[label_values != true_label]
    new_label = np.random.choice(incorrect_choices)
    
    df_poisoned.loc[idx, "species"] = new_label

print("Poisoned label indices:", poison_indices)
print(df_poisoned.loc[poison_indices, ["species"]])

filename = "data/poisoned_data" + str(int(poison_fraction * 100)) + ".csv"
df_poisoned.to_csv(filename, index=False)
print("poisoned data saved saved to "+filename)




