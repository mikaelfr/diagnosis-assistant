import math
import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from patient_generator import generate_people, generate_people2
from classifier import convert_for_classifier

people = generate_people2(100000)

# split into test and train datasets
split_idx = math.floor(len(people.index) * 0.8)
train_df = people.iloc[:split_idx]
test_df = people.iloc[split_idx:]

X_train, y_train = convert_for_classifier(train_df)
X_test, y_test = convert_for_classifier(test_df)

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')
