import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained('darrow312/finetunedHelsinkiMajorProject')
model = AutoModelForSeq2SeqLM.from_pretrained('darrow312/finetunedHelsinkiMajorProject')

# Define the translation function
def predict(sentence):
    inputs = tokenizer(sentence, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Define the Streamlit app
def main():
    st.title('Translate Sentences')
    sentence = st.text_area('Enter Sentence (English):')
    if st.button('Translate'):
        translation = predict(sentence)
        st.write('Translated Sentence (Hindi):', translation)

if __name__ == '__main__':
    main()
