import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")

def main():
    st.title("Llama 3 Chat")

    pipe = load_model()

    user_input = st.text_input("You:")
    if user_input:
        messages = [{"role": "user", "content": user_input}]
        response = pipe(messages)
        st.write("Llama 3:", response[0]['generated_text'])

if __name__ == "__main__":
    main()
