###############################
####Evaluation           ######
###############################

import pandas as pd

from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine,euclidean
from sklearn.metrics import euclidean_distances


def count_hits(col,k,dataset):
    topsim=list(col.sort_values()[0:k].index)
    s_label=dataset[topsim[0]]["label"]
    lab_list=[dataset[i]["label"] for i in topsim[1:]]
    return len([l for l in lab_list if l==s_label ])

def evaluate_model(dataset,vec_list,k=5):
    similarities = euclidean_distances(vec_list)
    sim=pd.DataFrame(similarities)
    a=sim.apply(lambda x:count_hits(x,k,dataset),axis=1)
    a.sum()
    result_score=a.sum()/(len(sim)*(k-1))
    return result_score
