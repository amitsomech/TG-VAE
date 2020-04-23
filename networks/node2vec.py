#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data
from utils.table2graph import create_corpus, shuffle_vocabulary, build_graph_edges, build_node_features
from utils.vectorize import generate_table_vectors
from utils.evaluation import evaluate_model

import utils.evaluation


def run_model(dataset,conf):
# ## 1) Build Table graph
# ### Tables tokenization
    tokenized_tables, vocabulary, cell_dict, reversed_dictionary = corpus_tuple = create_corpus(dataset,include_attr=conf["add_attr"])
    if conf["shuffle_vocab"] == True:
        shuffled_vocab = shuffle_vocabulary(vocabulary)
    else:
        shuffled_vocab = None

    nodes = build_node_features(vocabulary)
    row_edges_index, row_edges_weights = build_graph_edges(tokenized_tables,s_vocab=shuffled_vocab,sample_frac=conf["row_edges_sample"],columns=False)
    col_edges_index, col_edges_weights = build_graph_edges(tokenized_tables,s_vocab=shuffled_vocab,sample_frac=conf["column_edges_sample"],columns=True)

    edges = torch.cat((row_edges_index,col_edges_index),dim=1)
    weights= torch.cat((row_edges_weights,col_edges_weights),dim=0)
    graph_data = Data(x=nodes,edge_index=edges,edge_attr=weights)

    # ## 2 ) Run Table Auto-Encoder Model:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = DataLoader(torch.arange(graph_data.num_nodes), batch_size=128, shuffle=True)
    graph_data = graph_data.to(device)
    
    def train():
        model.train()
        total_loss = 0
        for subset in loader:
            optimizer.zero_grad()
            loss = model.loss(graph_data.edge_index, subset.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    model = Node2Vec(graph_data.num_nodes, embedding_dim=conf["vector_size"], walk_length=conf["n2v_walk_length"],
                     context_size=conf["n2v_context_size"], walks_per_node=conf["n2v_walks_per_node"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses=[]
    for epoch in range(conf["epoch_num"]):
        loss = train()
        print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss))
        losses.append(float(loss))
    # ### 3) Extract the latent cell vectors, generate table vectors:
    model.eval()
    with torch.no_grad():
        z = model(torch.arange(graph_data.num_nodes, device=device))
        cell_vectors = z.cpu().numpy()
    vec_list=generate_table_vectors(cell_vectors,tokenized_tables,s_vocab=shuffled_vocab)

    # ## 3) Evaluate the model
    result_score=evaluate_model(dataset,vec_list,k=5)
    return cell_vectors,vec_list,losses,result_score

