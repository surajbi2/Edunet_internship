import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

if not os.path.exists(nltk.data.find("tokenizers/punkt")):
    nltk.download("punkt")
if not os.path.exists(nltk.data.find("corpora/stopwords")):
    nltk.download("stopwords")

try:
    medical_qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")
except Exception as e:
    st.error(f"Error loading model: {e}")
    medical_qa_pipeline = None

def healthcare_chatbot(user_input):
    if medical_qa_pipeline is None:
        return "Model is not available. Please try again later."
    
    context = (
        "Healthcare is crucial for maintaining overall well-being. Common symptoms of flu include fever, cough, and sore throat. "
        "For serious medical conditions, it is advised to consult a certified doctor immediately. Medication should only be taken as prescribed by a healthcare professional. "
        "Appointments with specialists can be booked through official hospital websites or by contacting medical centers directly. Maintaining a healthy lifestyle includes balanced nutrition, regular exercise, and proper hydration. "
        "If you experience emergency symptoms such as severe chest pain or difficulty breathing, seek immediate medical attention. "
        "For cough relief, staying hydrated, drinking warm fluids, using a humidifier, and taking prescribed medication can help."
    )
    

    prompt = f"Based on the following context, answer the question: {user_input} Context: {context}"
    response = medical_qa_pipeline(prompt, max_length=150)[0]["generated_text"]
    return response


def main():
    st.title("AI-Powered Health Assistant: Reliable Health Information")
    st.write("Ask any medical or health-related question, and I will provide you with accurate and trustworthy information.")
    st.write("Caution: AI can make mistakes. Always consult a healthcare professional for medical advice.")
    user_input = st.text_input("Enter your question:", "")
    
    if st.button("Submit"):
        if user_input:
            response = healthcare_chatbot(user_input)
            st.write("**Healthcare Assistant:**", response)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
