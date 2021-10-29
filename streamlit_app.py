import math
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from patient_generator import generate_people, generate_people2, diagnoses_map
from classifier import classify_data, convert_for_classifier, classfier_col_order, fix_missing_values

def main():
    selected_page = st.sidebar.selectbox(
        'Select page',
        ('Landing', 'Example UI', 'Model', 'Patients'),
        index=0
    )

    if selected_page == 'Patients':
        patient_stats()
    elif selected_page == 'Model':
        model_stats()
    elif selected_page == 'Example UI':
        example_user_ui()
    else:
        landing()

def landing():
    st.title('Landing page')
    st.write('Select a page from the sidebar.')

def model_stats():
    st.title('Model Stats')

    with st.spinner('Running the model...'):
        people = generate_people2(100000)

        # split into test and train datasets
        people = people.sample(frac=1).reset_index(drop=True)
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
    patient_df = None
    # caching the classfier
    if 'patient_df' not in st.session_state:
        with st.spinner('Generating people...'):
            patient_df = generate_people2(100000)
            st.session_state['patient_df'] = patient_df
    else:
        patient_df = st.session_state['patient_df']

    st.title('Simulated Patient Data')
    st.write('Example of generated patient data of 100000 people:')
    should_reset = st.button('Regenerate patient data')
    if should_reset:
        patient_df = generate_people2(100000)
        st.session_state['patient_df'] = patient_df
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

    # Number of NAs for people with a diagnosis
    st.header('Number of NAs for people with a diagnosis')
    count_of_nas = patient_df[patient_df['diagnosis'] != 0].isnull().sum().T.reset_index()
    count_of_nas = count_of_nas.rename(columns={'index': 'column', 0: 'count'})
    nas_chart = alt.Chart(count_of_nas).mark_bar().encode(x='column', y='count')
    st.altair_chart(nas_chart, use_container_width=True)

    # Value distributions
    st.header('Value distributions per diagnosis')
    selected_diagnosis = st.selectbox(
        'Select diagnosis',
        ['all'] + list(diagnoses_map.keys()),
        index=0
    )

    continous = ['pulse', 'spo2', 'sap', 't', 'rr', 'gluc', 'age']
    categorical = ['sex', 'motor_impairment', 'chest_pain']

    if selected_diagnosis == 'all':
        dist_df = patient_df
    else:
        dist_df = patient_df[patient_df['diagnosis'] == diagnoses_map[selected_diagnosis]]

    for col in continous:
        df_max = dist_df[col].max()
        df_min = dist_df[col].min()
        num_bins = round(df_max - df_min)
        num_bins = num_bins * 10 if num_bins < 10 else num_bins
        num_bins = min(num_bins, 30)
        binned = pd.cut(dist_df[col], num_bins)
        value_counts = binned.value_counts()
        value_counts = value_counts.to_frame().reset_index()
        value_counts = value_counts.rename(columns={'index': 'value', col: 'count'})
        value_counts['value'] = value_counts['value'].map(lambda x: round(x.mid, 3))
        chart = alt.Chart(value_counts, title=col).mark_bar().encode(x='value', y='count')
        st.altair_chart(chart, use_container_width=True)


    for col in categorical:
        value_counts = dist_df[col].value_counts()
        value_counts = value_counts.to_frame().reset_index()
        value_counts = value_counts.rename(columns={'index': 'value', col: 'count'})
        if col == 'sex':
            value_counts['value'] = value_counts['value'].replace(to_replace={0: 'male', 1: 'female'})
        else:
            value_counts['value'] = value_counts['value'].replace(to_replace={0: 'no', 1: 'yes'})
        chart = alt.Chart(value_counts, title=col).mark_bar().encode(x='value', y='count')
        st.altair_chart(chart, use_container_width=True)


def example_user_ui():
    st.title('Example UI')

    classifier = None
    means = None
    # caching the classfier
    if 'precomputed_classifier' not in st.session_state:
        with st.spinner('Generating the model...'):
            people = generate_people2(100000)
            people = people.sample(frac=1).reset_index(drop=True)
            classifier, means = classify_data(people)
            st.session_state['precomputed_classifier'] = classifier
            st.session_state['means'] = means
    else:
        classifier = st.session_state['precomputed_classifier']
        means = st.session_state['means']

    with st.form('parameters_form'):
        continous = ['pulse', 'spo2', 'sap', 't', 'rr', 'gluc', 'age']
        categorical = ['motor_impairment', 'chest_pain']

        measurements = {}

        for val in continous:
            # assuming 0 is null
            num_val = st.number_input(val)
            measurements[val] = num_val if num_val != 0 else None

        measurements['sex'] = 0 if st.selectbox('sex', ('Male', 'Female')) == 'Male' else 1

        for val in categorical:
            measurements[val] = int(st.checkbox(val))

        submitted = st.form_submit_button('Get probabilities')
        if submitted:
            # get probabilities
            df = pd.DataFrame(data=[measurements])
            # we need to use means from the training set here, kinda dangerous
            # but should be fine medical measurements in this case
            df = fix_missing_values(df, means)
            df = df[list(filter(lambda x: x != 'diagnosis', classfier_col_order))]
            probabilities = classifier.predict_proba(df.to_numpy())

            # build graph
            diagnoses_map_flipped = { v: k for k,v in diagnoses_map.items()}
            diagnoses_df = pd.DataFrame(data={'diagnosis': range(len(probabilities[0])), 'probability': probabilities[0]})
            diagnoses_df['diagnosis'] = diagnoses_df['diagnosis'].replace(to_replace=diagnoses_map_flipped)
            diagnoses_chart = alt.Chart(diagnoses_df).mark_bar().encode(x='diagnosis', y='probability')
            st.altair_chart(diagnoses_chart, use_container_width=True)

main()
