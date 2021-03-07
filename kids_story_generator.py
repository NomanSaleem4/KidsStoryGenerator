import streamlit as st
home_dir = "/home/noman/Downloads/kids_model/"
# home_dir = "/content/drive/My Drive/Colab Notebooks/Kids_Story_Generation/"

from aitextgen import aitextgen


st.title('Kids Story Generator by DeepLearningPro')

# model_dir = "https://drive.google.com/drive/u/2/folders/17KR0I2-RXJkruEJfZA3GfWCtzTEEvybk"
model_dir = "{}{}".format(home_dir, "model/")



# import gdown

# url = 'https://drive.google.com/drive/folders/17KR0I2-RXJkruEJfZA3GfWCtzTEEvybk?usp=sharing'
# output = '/home/noman/Documents/KidsStoryGenerator/model'
# gdown.download(url, output, quiet=False) 




# file_id = 'https://drive.google.com/drive/folders/17KR0I2-RXJkruEJfZA3GfWCtzTEEvybk?usp=sharing'
# from google.colab import files
# files.download("https://drive.google.com/drive/folders/17KR0I2-RXJkruEJfZA3GfWCtzTEEvybk?usp=sharing")



ai = aitextgen(model=model_dir+"pytorch_model.bin", config=model_dir+"config.json", to_gpu=False)



desc = "Pre-trained GPT2 model fine tuned on the Kids Stories"
st.write(desc)
story_length = st.number_input('Story length in words', min_value=1, max_value=150, value=30)
seed_text = st.text_input('Seed Text')

if st.button('Generate Text'):
    generated_text = ai.generate_one(prompt=seed_text, max_length=story_length)
    st.write(generated_text)

