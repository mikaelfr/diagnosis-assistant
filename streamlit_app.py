import math
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from patient_generator import generate_people, diagnoses_map
from classifier import classify_data, convert_for_classifier

def main():
    selected_page = st.sidebar.selectbox(
        'Select page',
        ('Landing', 'Model', 'Patients'),
        index=0
    )

    if selected_page == 'Patients':
        patient_stats()
    elif selected_page == 'Model':
        model_stats()
    else:
        landing()

def landing():
    st.title('Landing page')
    st.write('Select a page from the sidebar.')

def model_stats():
    st.title('Model Stats')

    with st.spinner('Running the model...'):
        people = generate_people(100000)

        # split into test and train datasets
        split_idx = math.floor(len(people.index) * 0.8)
        train_df = people.iloc[:split_idx]
        test_df = people.iloc[split_idx:]

        classifier = classify_data(train_df)

        # check accuracy
        samples, features = convert_for_classifier(test_df)
        predicted_features_prob = classifier.predict_proba(samples)
        predicted_features = np.argmax(predicted_features_prob, axis=1)

        correct = features == predicted_features
        unique, counts = np.unique(correct, return_counts=True)
        correct_counts = dict(zip(unique, counts))
        accuracy = correct_counts[True] / (correct_counts[True] + correct_counts[False])
        st.write('Accuracy: ', accuracy)

        # Print out some wrong predictions
        test_df['predicted'] = predicted_features
        test_df['predicted_conf'] = np.amax(predicted_features_prob, axis=1)

        mean_conf_correct = test_df[correct]['predicted_conf'].mean()
        mean_conf_not_correct = test_df[~correct]['predicted_conf'].mean()

        st.write('Mean confidence for correct predictions: ', mean_conf_correct)
        st.write('Mean confidence for false predictions: ', mean_conf_not_correct)

        st.header('Wrong predictions')
        st.write('Some examples of wrong predictions')
        st.write(test_df[~correct].head(10))

def patient_stats():
    patient_df = generate_people(100000)
    st.title('Simulated Patient Data')
    st.write('Example of generated patient data of 100000 people:')
    st.write(patient_df.head())

    # Group by diagnoses
    st.header('Number of diagnoses')
    diagnoses_map_flipped = { v: k for k,v in diagnoses_map.items()}
    count_of_diagnoses = patient_df.groupby('diagnosis').size().reset_index(name='counts')
    count_of_diagnoses['diagnosis'] = count_of_diagnoses['diagnosis'].replace(to_replace=diagnoses_map_flipped)
    diagnoses_chart = alt.Chart(count_of_diagnoses).mark_bar().encode(x='diagnosis', y='counts')
    st.altair_chart(diagnoses_chart, use_container_width=True)

    # Number of NAs
    st.header('Number of NAs')
    count_of_nas = patient_df.isnull().sum().T.reset_index()
    count_of_nas = count_of_nas.rename(columns={'index': 'column', 0: 'count'})
    nas_chart = alt.Chart(count_of_nas).mark_bar().encode(x='column', y='count')
    st.altair_chart(nas_chart, use_container_width=True)

    # Value distributions
    st.header('Value distributions')
    continous = ['pulse', 'spo2', 'sap', 't', 'rr', 'gluc', 'age']
    categorical = ['sex', 'motor_impairment', 'chest_pain']

    for col in continous:
        df_max = patient_df[col].max()
        df_min = patient_df[col].min()
        num_bins = round(df_max - df_min)
        num_bins = num_bins * 10 if num_bins < 10 else num_bins
        num_bins = min(num_bins, 30)
        binned = pd.cut(patient_df[col], num_bins)
        value_counts = binned.value_counts()
        value_counts = value_counts.to_frame().reset_index()
        value_counts = value_counts.rename(columns={'index': 'value', col: 'count'})
        value_counts['value'] = value_counts['value'].map(lambda x: round(x.mid, 3))
        chart = alt.Chart(value_counts, title=col).mark_bar().encode(x='value', y='count')
        st.altair_chart(chart, use_container_width=True)


    for col in categorical:
        value_counts = patient_df[col].value_counts()
        value_counts = value_counts.to_frame().reset_index()
        value_counts = value_counts.rename(columns={'index': 'value', col: 'count'})
        if col == 'sex':
            value_counts['value'] = value_counts['value'].replace(to_replace={0: 'male', 1: 'female'})
        else:
            value_counts['value'] = value_counts['value'].replace(to_replace={0: 'no', 1: 'yes'})
        chart = alt.Chart(value_counts, title=col).mark_bar().encode(x='value', y='count')
        st.altair_chart(chart, use_container_width=True)


main()
