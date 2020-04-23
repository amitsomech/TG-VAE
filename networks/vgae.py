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

    x, train_pos_edge_index = nodes,edges
    
    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(Encoder, self).__init__()
            self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_logvar = GCNConv(
                2 * out_channels, out_channels, cached=True)
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


    channels=conf["vector_size"]
    enc = Encoder(graph_data.num_features, channels)
    model = VGAE(enc)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train(model,optimizer,x, train_pos_edge_index):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        rl = model.recon_loss(z, train_pos_edge_index)
        kl = model.kl_loss()

        loss=rl+kl

        loss.backward()
        optimizer.step()
        return (rl,kl,loss)




    losses=[]
    for epoch in range(conf["epoch_num"]):
        loss = train(model,optimizer,x,train_pos_edge_index)
        losses.append(loss)
        print(epoch,loss)
        losses.append(loss)
    # ### 3) Extract the latent cell vectors, generate table vectors:
    def get_cell_vectors(model,x,train_pos_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
            cell_vectors = z.numpy()
        return z,cell_vectors

    z,cell_vectors = get_cell_vectors(model,x,train_pos_edge_index)





    vec_list=generate_table_vectors(cell_vectors,tokenized_tables,s_vocab=shuffled_vocab)

    # ## 3) Evaluate the model
    result_score=evaluate_model(dataset,vec_list,k=5)
    return cell_vectors,vec_list,losses,result_score

