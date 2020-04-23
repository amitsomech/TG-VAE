###############################
####Table2Graph          ######
###############################

import torch
import numpy as np
import pandas as pd
import os


####  Table Tokenization ######

def increment_count(cell_dict, ID, key):
    if key in cell_dict:        
        cell_dict[key] = (cell_dict[key][0], cell_dict[key][1] + 1)
    else:
        cell_dict[key] = (ID, 1)        
        ID += 1
    return ID, cell_dict[key][0]


def tokenize_table(table,cell_dict,ID,include_attr=True):
    ttable=np.zeros(table.shape,dtype=np.int64)
    for i,row in enumerate(table.values):
        for (j, cell) in enumerate(row):
            if include_attr:
                key=(table.columns[j],cell)
            else:
                key=cell
            ID, curr_ID  = increment_count(cell_dict, ID,key)             
            ttable[i,j]=curr_ID
    return ttable,ID


def create_corpus(tables,include_attr=False):
    cell_dict={}
    ID=0
    tokenized_tables=[]
    for table_entry in tables:
        table=table_entry["table"]
        tokenized_tt,ID= tokenize_table(table,cell_dict,ID,include_attr=include_attr)
        tokenized_tables.append(tokenized_tt)
    vocabulary=range(ID)
    reversed_dictionary = dict(zip([x[0] for x in cell_dict.values()], cell_dict.keys()))
    
    return tokenized_tables, vocabulary, cell_dict, reversed_dictionary


####    Build Graph      ######

def shuffle_vocabulary(vocabulary):
    return np.random.permutation(vocabulary)


def process_row(edges_dict,row,s_vocab=None):
    for r1 in row:
        for r2 in row:
            if r1==r2:
                continue
            s,t = tuple(sorted((r1,r2)))
            if s_vocab is not None:
                s,t = s_vocab[s],s_vocab[t]
            if ((s,t)) in edges_dict:
                edges_dict[(s,t)]+=1
            else:
                edges_dict[(s,t)]=1

def get_edges(matrices,s_vocab=None,columns=False):
    edges_dict={}
    for m in matrices:
        if columns:
            m=m.T
        for row in m:
            process_row(edges_dict,row,s_vocab=s_vocab)
    return edges_dict
    
def build_graph_edges(tokenized_tables,s_vocab=None,sample_frac=1.0,columns=False):
    edges=get_edges(tokenized_tables,s_vocab=s_vocab,columns=columns)
    edges_list = []
    for (s,t), size in edges.items():
        edges_list.append({"source":s,"target":t,"weight":size})
    edges_df = pd.DataFrame(edges_list)
    #take sample:
    sample_edges=edges_df.sample(frac=sample_frac,weights='weight')
    
    #build the coo edges arrays:
    sources=sample_edges.source.values
    targets=sample_edges.target.values
    e1=np.concatenate((sources,targets))
    e2=np.concatenate((targets,sources))
    weights = sample_edges.weight.values
    edge_index = torch.tensor([e1,e2],dtype=torch.long)
    edge_weights= torch.tensor(np.concatenate((weights,weights)),dtype=torch.float32)
    return edge_index, edge_weights

def build_node_features(vocabulary,type="ones"):
    #ways to deal with **featuresless** nodes
    if type == "ones":
        nodes = torch.ones(len(vocabulary),dtype=torch.float32)
        nodes=nodes.reshape((len(nodes),1))
    elif type == "ordinal":
        nodes=torch.tensor(vocabulary,dtype=torch.float32)
        nodes=nodes.reshape((len(nodes),1))
    elif type == "onehot":
        nodes = torch.sparse.torch.eye(len(vocabulary))
    return nodes
        







