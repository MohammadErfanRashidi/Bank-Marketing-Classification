import streamlit as st
import joblib
import numpy as np
import pandas as pd

@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.pkl')
    classifier = joblib.load('models/model.pkl')
    return scaler, classifier

scaler, classifier = load_models()

st.title('Bank Subscription Predictor')
st.markdown('Predict whether a client will subscribe or not.')

tab_single, tab_batch = st.tabs(['Single Prediction', 'Batch Prediction (CSV)'])

with tab_single:
    st.subheader('Enter Client Information')
    col1, col2 = st.columns(2)

    with col1:

        poutcome = st.selectbox(
            'Previous Campaign Outcome',
            options = ['success', 'failure', 'other'],
            index = 0
        )

        month = st.selectbox(
            'Last Contact Month',
            options = ['jan', 'feg', 'mar', 'apr', 'may', 'jun',
                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
            index = 2
        )

        pdays = st.number_input(
            'Days Since Last Contact (-1 of never contacted)',
            min_value = -1, value = -1, step = 1, max_value = 999
        )

        contact = st.selectbox(
            'Contact Type',
            options = ['cellular', 'telephone', 'unknown']
        )

    with col2:

        housing = st.radio(
            'Has Housing Loan?',
            options = ['yes', 'no'],
            horizontal = True
        )

        job = st.selectbox(
            'Job Type',
            options = ['student', 'retired', 'other'],
            index = 0
        )

        duration = st.number_input(
            'Last Contact Duration (seconds)',
            min_value = 0, values = 100, step = 1
        )

    pdays_replaced_feat = 999 if pdays == -1 else pdays

    poutcome_success_feat = 1 if poutcome == 'success' else 0

    month_dec_oct_sep_mar_feat = 1 if month in ['dec', 'oct', 'sep', 'mar'] else 0

    contact_cellular_telephone_feat = 1 if contact in ['cellular', 'telephone'] else 0

    housing_feat = 1 if housing == 'yes' else 0

    job_student_or_retired_feat = 1 if job in ['student', 'retired'] else 0

    features = np.array([
        duration,
        poutcome_success_feat,
        month_dec_oct_sep_mar_feat, 
        pdays_replaced_feat,
        contact_cellular_telephone_feat,
        housing_feat,
        job_student_or_retired_feat
    ]).reshape(1, -1)

    to_scale = np.array([[features[0, 3], features[0, 0]]])
    scaled_values = scaler.transform(to_scale)
    features[0, 3] = scaled_values[0, 0]
    features[0, 0] = scaled_values[0, 1]

    prediction = classifier.predict(features)[0]
    proba = classifier.predict_proba(features)[0, 1]

    st.markdown('---')
    if prediction == 1:
        st.success('The client is likely to **subscribe**')
    else:
        st.error('The client is likely **not to subscribe**')
    st.info(f'Predicted probability of subscription: {proba:.2f}')



with tab_batch:
    st.subheader('Upload a CSV file')
    st.markdown('''
    **Required Columns:**
    `poutcome`, `month`, `pdays`, `contact`, `housing`, `job`, `duration`
    (Column names must match)
    ''')

    uploaded_file = st.file_uploader('Choose a CSV file', type = 'csv')

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f'Could not read file: {e}')
            st.stop()
        
        required = ['poutcome', 'month', 'pdays', 'contact', 'housing', 'job', 'duration']
        missing = [col for col in required if col not in df_input.columns]

        if missing:
            st.error(f'Missing columns: {', '.join(missing)}')
            st.stop()

        df_out = df_input.copy()

        df_out['poutcome_success_feat'] = df_out['poutcome'].isin(['success']).astype
        df_out['month_dec_oct_sep_mar_feat'] = df_out['month'].isin(['dec', 'oct', 'sep', 'mar'])
        df_out['pdays_replaced_feat'] = df_out['pdays'].replace(-1, 999)
        df_out['contact_cellular_telephone_feat'] = df_out['contact'].isin(['cellular', 'telephone']).astype(int)
        df_out['housing_feat'] = df_out['housing'].map({'yes': 1, 'no': 0})
        df_out['job_student_or_retired_feat'] = df_out['job'].isin(['student', 'retired']).astype(int)

        feature_cols = [
            'duration',
            'poutcome_success_feat',
            'month_dec_oct_sep_mar_feat',
            'pdays_replaced_feat',
            'contact_cellular_telephone_feat',
            'housing_feat',
            'job_student_or_retired_feat'
        ]

        X = df_out[feature_cols].values

        scale_features = X[:, [3, 0]]
        scaled = scaler.transform(scale_features)
        X[:, 3] = scaled[:, 0]
        X[:, 0] = scaled[:, 1]

        y_pred = classifier.predict(X)
        df_out['y'] = y_pred
        df_out['probability'] = classifier.predict_proba(X)[:, 1]

        st.markdown('### Prediction results (first top 10 rows)')
        st.dataframe(df_out.head(10))

        csv = df_out.to_csv(index = False).encode('utf-8')
        st.download_button(
            label = 'Download CSV with predictions',
            data = csv,
            file_name = 'prediction.csv',
            mime = 'text/csv'
        )