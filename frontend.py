import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="Prédiction du churn", page_icon="📊")
st.title("Prédiction du risque de départ")
st.caption("Renseignez le profil du client pour obtenir une prédiction.")

with st.form("churn_prediction"):
    left, right = st.columns(2)

    with left:
        credit_score = st.number_input(
            "Score de crédit", min_value=300, max_value=850, value=619, step=1
        )
        gender = st.selectbox("Genre", ["Female", "Male"])
        age = st.number_input("Âge", min_value=18, max_value=100, value=42, step=1)
        tenure = st.number_input("Ancienneté (années)", min_value=0, max_value=10, value=2, step=1)
        balance = st.number_input("Solde", min_value=0.0, value=0.0, step=100.0)

    with right:
        products = st.number_input(
            "Nombre de produits", min_value=1, max_value=4, value=1, step=1
        )
        has_card = st.selectbox(
            "Possède une carte",
            options=[0, 1],
            format_func=lambda value: "Non" if value == 0 else "Oui",
        )
        active_member = st.selectbox(
            "Membre actif",
            options=[0, 1],
            format_func=lambda value: "Non" if value == 0 else "Oui",
        )
        salary = st.number_input(
            "Salaire estimé", min_value=0.0, value=101348.88, step=100.0
        )

    submitted = st.form_submit_button("Prédire le risque")

if submitted:
    payload = {
        "CreditScore": credit_score,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": active_member,
        "EstimatedSalary": salary,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
    except requests.HTTPError as error:
        detail = error.response.text if error.response is not None else str(error)
        st.error(f"L'API a retourné une erreur : {detail}")
    except requests.RequestException as error:
        st.error(f"Impossible de contacter l'API à {API_URL} : {error}")
    else:
        result = response.json()
        if result["prediction"] == 1:
            st.warning(result["interpretation"])
        else:
            st.success(result["interpretation"])
        st.metric(
            "Probabilité de churn",
            f"{result['probability_churn']:.1%}",
        )
