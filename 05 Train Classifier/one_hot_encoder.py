import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

metadata_df = pd.read_csv("../Datasets/dataset/train.csv", sep=',')

sorted_sex = sorted(metadata_df.sex.unique())
sorted_localization = sorted(metadata_df.localization.unique())
sorted_dx = sorted(metadata_df.dx.unique())

print("sex", sorted_sex)
print("localization", sorted_localization)
print("dx", sorted_dx)

def one_hot_encode_sex(sex):
    sex_vector = np.array([])
    for s in sorted_sex:
        if s == sex:
            sex_vector = np.append(sex_vector, 1)
        else:
            sex_vector = np.append(sex_vector, 0)
    return sex_vector

def one_hot_encode_localization(localization):
    localization_vector = np.array([])
    for s in sorted_localization:
        if s == localization:
            localization_vector = np.append(localization_vector, 1)
        else:
            localization_vector = np.append(localization_vector, 0)
    return localization_vector


def one_hot_encode_dx(dx):
    dx_vector = np.array([])
    for s in sorted_dx:
        if s == dx:
            dx_vector = np.append(dx_vector, 1)
        else:
            dx_vector = np.append(dx_vector, 0)
    return dx_vector
    