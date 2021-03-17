import streamlit as st
from aitextgen import aitextgen
import Algorithmia

# home_dir = "/home/noman/Downloads/kids_model/"
# model_dir = "{}{}".format(home_dir, "model/")
# ai = aitextgen(model=model_dir+"pytorch_model.bin", config=model_dir+"config.json", to_gpu=False)

client = Algorithmia.client('sim4hQ7vkovFU2ffnS275i2allq1')
algo = client.algo('nomansaleem92/stories/0.1.0')
# algo.set_options(timeout=300) # optional


st.title('Kids Story Generator by DeepLearningPro')
desc = "Pre-trained GPT2 model fine tuned on the Kids Stories"
st.write(desc)
story_length = st.number_input('Story length in words', min_value=1, max_value=250, value=30)
seed_text = st.text_input('Seed Text')

if st.button('Generate Text'):
    # generated_text = ai.generate_one(prompt=seed_text, max_length=story_length)
    input = {"seed_text": seed_text, "story_length": story_length}
    generated_text = algo.pipe(input).result
    st.write(generated_text)

