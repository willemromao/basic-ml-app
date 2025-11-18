import streamlit as st
import requests

st.set_page_config(page_title="Intent Classifier", page_icon="🤖")

st.title("🤖 Intent Classifier")

# Config
API_URL = "http://localhost:8000"

# Sidebar
with st.sidebar:
    model = st.selectbox("Modelo", ["confusion-clf", "clair-clf"])
    owner = st.text_input("Owner", value="demo_user")

# Input
text_input = st.text_area("Digite o texto:", height=100)

# Predict button
if st.button("Classificar", type="primary"):
    if text_input.strip():
        with st.spinner("Classificando..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    params={"text": text_input, "owner": owner, "load_model": model},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    model_result = result["predictions"][model]
                    
                    # Resultado
                    st.success(f"**Intent:** {model_result['top_intent']}")
                    
                    # Probabilidades
                    st.write("**Probabilidades:**")
                    for intent, prob in sorted(model_result['all_probs'].items(), key=lambda x: x[1], reverse=True):
                        st.progress(prob, text=f"{intent}: {prob:.1%}")
                else:
                    st.error(f"Erro: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Erro: {e}")
    else:
        st.warning("Digite algum texto!")