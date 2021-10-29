import math
import pandas as pd
import numpy as np
from patient_generator import generate_people, generate_people2
from classifier import classify_data, convert_for_classifier

people = generate_people2(100000)

# split into test and train datasets
split_idx = math.floor(len(people.index) * 0.8)
train_df = people.iloc[:split_idx]
test_df = people.iloc[split_idx:]

classifier, _ = classify_data(train_df)

# check accuracy
samples, features = convert_for_classifier(test_df)
predicted_features_prob = classifier.predict_proba(samples)
predicted_features = np.argmax(predicted_features_prob, axis=1)

correct = features == predicted_features
unique, counts = np.unique(correct, return_counts=True)
correct_counts = dict(zip(unique, counts))
accuracy = correct_counts[True] / (correct_counts[True] + correct_counts[False])
print('Accuracy:', accuracy)

# Print out some wrong predictions
test_df['predicted'] = predicted_features
test_df['predicted_conf'] = np.amax(predicted_features_prob, axis=1)
print(test_df[~correct].head(10))
