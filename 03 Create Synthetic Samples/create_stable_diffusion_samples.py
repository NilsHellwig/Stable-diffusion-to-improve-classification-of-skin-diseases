from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import shutil
import os
import torch
from diffusers import StableDiffusionPipeline

DATASET_METADATA_PATH = "../Datasets/raw_dataset.csv"
RAW_DATASET_PATH = "../Datasets/raw_dataset/"
DATASET_PATH = "../Datasets/dataset/"
SYNTH_DATASET_PATH = "../Datasets/synth_dataset/"
SEED = 42
N_EXAMPLES_FOR_LABEL = 1000
MODEL_PATH = "/mnt/data/stable_diffusion_2_skin_balanced/"
OUTPUT_CSV = "../Datasets/generative_prompts.csv"


LABEL_PROMPTS = {
                 "akiec": "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
                 "bcc":"basal cell carcinoma",
                 "df": "dermatofibroma",
                 "mel": "melanoma",
                 "nv": "melanocytic nevi",
                 "vasc": "vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)",
                 "bkl": "benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)"
                }

np.random.seed(SEED)
random.seed(SEED)

pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, use_auth_token=True, safety_checker = None)
pipe.to("cuda")

train_df = pd.read_csv(DATASET_PATH + "train.csv")

df_generative_prompts = pd.DataFrame({
    'file_name': [], 
    'text': [],
    'localization': [],
    'sex': [],
    'age': [],
    'dx': []
})

def get_random_value(df, column_name):
    column = df[column_name]
    #print(column)
    random_value = np.random.choice(column)
    return random_value

vc = train_df["dx"].value_counts()

for key, value in vc.items():
    vc[key] = 1000

print(vc)

def generate_image(file_name, prompt, main_label):
    try:
        os.makedirs(SYNTH_DATASET_PATH + main_label)
    except FileExistsError:
        pass
    
    images = pipe(prompt=prompt, num_images_per_prompt=16, height=512, width=512, guidance_scale=3).images
    itr = 0
    for image in images:
        image.save(SYNTH_DATASET_PATH + file_name + "_" + str(itr) + ".jpg")
        itr += 1
        
for key, value in vc.items():
    for i in range(0, value):
        df_dx = train_df[train_df["dx"] == key]
        sex = get_random_value(df_dx, "sex")
        localization = get_random_value(df_dx, "localization")
        age = get_random_value(df_dx, "age")
        
        prompt = LABEL_PROMPTS[key] + " " + sex + " " + localization + " " + str(age)
        file_name = key + "/" + key + "_" + str(i)
        
        new_row = {'file_name': file_name, 'text': prompt, 'sex': sex, 'localization': localization, 'age': age, 'dx': key}
        generate_image(file_name, prompt, key)
            
        df_generative_prompts = pd.concat([df_generative_prompts, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        
df_generative_prompts.to_csv(OUTPUT_CSV)