###############################
####Vectorize            ######
###############################

import numpy as np

def get_vector(cell_id,cell_vectors,s_vocab=None):
    if s_vocab is not None:
        cell_id = s_vocab[cell_id]
    return cell_vectors[cell_id]

def generate_table_vectors(cell_vectors,tokenized_tables,s_vocab=None):
    vec_list=[]
    for table in tokenized_tables:
        vec_table=[get_vector(x,cell_vectors,s_vocab=s_vocab) for _,x in np.ndenumerate(table)]
        m=np.nansum(vec_table,axis=0)
        vec_list.append(m)
    return vec_list    
