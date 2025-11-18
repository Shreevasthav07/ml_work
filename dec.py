import pandas as pd
import numpy as np
from math import log2

# Step 1: Load dataset
df = pd.read_csv('play.csv')
df = df.drop('Day',axis=1)   # use your dataset
print("Dataset:\n", df, "\n")

# Step 2: Entropy function
def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    ent = 0
    for i in range(len(values)):
        p = counts[i]/np.sum(counts)
        ent -= p * log2(p)
    return ent

# Step 3: Information Gain
def info_gain(data, feature, target):
    total_entropy = entropy(data[target])
    vals, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = 0
    for i in range(len(vals)):
        subset = data[data[feature] == vals[i]]
        weighted_entropy += (counts[i]/np.sum(counts)) * entropy(subset[target])
    gain = total_entropy - weighted_entropy
    return gain

# Step 4: ID3 Algorithm (Recursive)
def id3(data, target, features):
    # if all targets are same
    if len(np.unique(data[target])) == 1:
        return np.unique(data[target])[0]
    # if no more features
    if len(features) == 0:
        return data[target].mode()[0]
    
    # choose feature with max info gain
    gains = [info_gain(data, f, target) for f in features]
    best_feature = features[np.argmax(gains)]
    tree = {best_feature: {}}
    
    # create subtrees
    for value in np.unique(data[best_feature]):
        sub_data = data[data[best_feature] == value]
        sub_features = [f for f in features if f != best_feature]
        subtree = id3(sub_data, target, sub_features)
        tree[best_feature][value] = subtree
    return tree

# Step 5: Build the tree
features = list(df.columns[:-1])  # all except target
target = df.columns[-1]
tree = id3(df, target, features)
print("\nDecision Tree:\n", tree)

# Step 6: Prediction
def predict(sample, tree):
    for key in tree.keys():
        value = sample[key]
        tree_val = tree[key][value]
        if isinstance(tree_val, dict):
            return predict(sample, tree_val)
        else:
            return tree_val

# Step 7: Evaluate
predictions = []
for i in range(len(df)):
    sample = df.iloc[i, :-1].to_dict()
    predictions.append(predict(sample, tree))

accuracy = np.mean(predictions == df[target])
print("\nPredictions:", predictions)
print("Accuracy:", round(accuracy * 100, 2), "%")


