import streamlit as st
from langchain.prompts import PromptTemplate

# # Load model directly
# from transformers import AutoModel
# # Use a pipeline as a high-level helper
# from transformers import pipeline

from langchain.llms.ctransformers import CTransformers
from langchain import HuggingFaceHub
from dotenv import load_dotenv
load_dotenv()

def generate_blog(topic,n_words,style):

    # model = AutoModel.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")
    # # pipe = pipeline("text-generation", model="TheBloke/Llama-2-7B-Chat-GGML")

    # Fetches LLM from Hugging Face Model Hub, after downloading it infers locally
    # Using pipeline also downloads the model and infers locally
    
    llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML",
                        model_file="llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type='llama',
                        config={'max_new_tokens': 256, 'repetition_penalty': 1.1, 'temperature':0.01},
                        )
    
    # Loading local LLM and inferencing locally
    # llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
    #                     model_type='llama',
    #                     config={'max_new_tokens': 256, 'repetition_penalty': 1.1, 'temperature':0.01}
    #                     )
    
    # llm = HuggingFaceHub(repo_id = "TheBloke/Llama-2-7B-Chat-GGML")
    

    mytemplate = f'''

    Write a blog for {style} profession on topic {topic} within {n_words} words.

    '''

    prompt = PromptTemplate(input_variables=['style', 'topic ', 'n_words'],
                            template=mytemplate)


    # format the prompt to add variable values
    prompt_formatted_str: str = prompt.format(
        style=style,
        topic=topic,
        no_words=n_words)

    # make a prediction
    prediction = llm.predict(prompt_formatted_str)
    print(prediction)
    return prediction
    
    

st.set_page_config(page_title="BLOG Generator", page_icon='🤖',layout='centered',initial_sidebar_state='collapsed')

st.header("BLOG Generation")

topic = st.text_input("Enter the topic on which you want to generate blog")

c1,c2 = st.columns([5,5])

with c1:
    n_words = st.text_input("No. of words")
    
with c2:
    style = st.selectbox("Writing Blog for ???",("Researchers","Data Scientists","Software Developers","Common People"))



submit = st.button("Submit")

if submit:
    response = generate_blog(topic,n_words,style)
    st.text_area("ANSWER",value=response)

