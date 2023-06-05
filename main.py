from inference import Inference
import streamlit as st

the_inference = Inference()

st.title("TREK GURU")

url = "https://en.wikivoyage.org/wiki/Trekking_in_Nepal"

st.write(f"The LLM is trained using this [data]({url})")

question = st.text_input('Your Question related to Trekking in Nepal')

if len(question) > 10:
    answer = the_inference.predict_answer(question)
else:
    answer = ""

st.write(answer)