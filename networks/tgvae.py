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
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)


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

    all_row_edges_index, all_row_edges_weights = build_graph_edges(tokenized_tables,s_vocab=shuffled_vocab,sample_frac=1.0,columns=False)
    all_col_edges_index, all_col_edges_weights = build_graph_edges(tokenized_tables,s_vocab=shuffled_vocab,sample_frac=1.0,columns=True)
    all_possible_edges= torch.cat((all_row_edges_index,all_col_edges_index),dim=1)

    edges = torch.cat((row_edges_index,col_edges_index),dim=1)
    weights= torch.cat((row_edges_weights,col_edges_weights),dim=0)
    graph_data = Data(x=nodes,edge_index=edges,edge_attr=weights)

    # ## 2 ) Run Table Auto-Encoder Model:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = DataLoader(torch.arange(graph_data.num_nodes), batch_size=128, shuffle=True)
    graph_data = graph_data.to(device)

    x, train_pos_edge_index = nodes,edges
    
    EPS = 1e-15
    MAX_LOGVAR = 10

    class TVGAE(GAE):
        r"""The Variational Graph Auto-Encoder model from the
        `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
        paper.

        Args:
            encoder (Module): The encoder module to compute :math:`\mu` and
                :math:`\log\sigma^2`.
            decoder (Module, optional): The decoder module. If set to :obj:`None`,
                will default to the
                :class:`torch_geometric.nn.models.InnerProductDecoder`.
                (default: :obj:`None`)
        """
        def __init__(self, encoder, decoder=None):
            super(TVGAE, self).__init__(encoder, decoder)

        def reparametrize(self, mu, logvar):
            if self.training:
                return mu + torch.randn_like(logvar) * torch.exp(logvar)
            else:
                return mu


        def encode(self, *args, **kwargs):
            """"""
            self.__rmu__, self.__rlogvar__,self.__cmu__, self.__clogvar__ = self.encoder(*args, **kwargs)
            self.__rlogvar__ = self.__rlogvar__.clamp(max=MAX_LOGVAR)
            self.__clogvar__ = self.__clogvar__.clamp(max=MAX_LOGVAR)
            zr = self.reparametrize(self.__rmu__, self.__rlogvar__)
            zc = self.reparametrize(self.__cmu__, self.__clogvar__)
            z=torch.cat((zr,zc),0)
            return z


        def kl_loss(self):

            rmu = self.__rmu__ 
            rlogvar = self.__rlogvar__ 

            cmu = self.__cmu__ 
            clogvar = self.__clogvar__ 
            
            rkl= -0.5 * torch.mean(
                torch.sum(1 + rlogvar - rmu**2 - rlogvar.exp(), dim=1))
            ckl= -0.5 * torch.mean(
                torch.sum(1 + clogvar - rmu**2 - clogvar.exp(), dim=1))
            return(rkl,ckl)

            
        def recon_loss(self,z, pos_edge_index,all_possible_edges):
            EPS = 1e-15
            MAX_LOGVAR = 10

            pos_loss = -torch.log(
                model.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

            # Do not include self-loops in negative samples
            pos_edge_index, _ = remove_self_loops(pos_edge_index)
            pos_edge_index, _ = add_self_loops(pos_edge_index)

            neg_edge_index = negative_sampling(all_possible_edges, z.size(0))
            neg_loss = -torch.log(1 -
                                  model.decoder(z, neg_edge_index, sigmoid=True) +
                                  EPS).mean()

            return pos_loss + neg_loss

    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(Encoder, self).__init__()
            self.conv_rows = GCNConv(in_channels, 2 * out_channels, cached=True)
            self.conv_cols = GCNConv(in_channels, 2 * out_channels, cached=True)
            
            self.conv_rmu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_rlogvar = GCNConv(2 * out_channels, out_channels, cached=True)

            self.conv_cmu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_clogvar = GCNConv(2 * out_channels, out_channels, cached=True)
                 
            
        def forward(self, x, row_edge_index,col_edge_index):
            xr = F.relu(self.conv_rows(x, row_edge_index))
            xc = F.relu(self.conv_cols(x, col_edge_index))
            return self.conv_rmu(xr, row_edge_index),\
                self.conv_rlogvar(xr, row_edge_index),\
                self.conv_cmu(xc, col_edge_index),\
                self.conv_clogvar(xc, col_edge_index)
        
            
        


    channels=conf["vector_size"]
    
    enc = Encoder(graph_data.num_features, channels)
    model = TVGAE(enc)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    def train(model,optimizer,x, row_edges,col_edges):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, row_edges,col_edges)
        mid = int(len(z)/2)
        zr=z[:mid]
        zc=z[mid:]
        
        #recon loss:
        rrl = model.recon_loss(zr,row_edges,all_possible_edges)
        crl = model.recon_loss(zc,col_edges,all_possible_edges)
        #loss = rrl+crl
        
        rkl,ckl = model.kl_loss()
        #loss = rkl+ckl
        
        loss = rrl+crl+rkl+ckl
        

        
        loss.backward()
        optimizer.step()
        #return loss,rrl,crl
        return loss,rrl,crl,rkl,ckl
    
    def get_cell_vectors(model,x,row_edges_index,col_edges_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, row_edges_index,col_edges_index)
            cell_vectors = z.numpy()
        return z,cell_vectors



    losses=[]
    results=[]
    for epoch in range(conf["epoch_num"]):
        #loss,row_loss,col_loss = train(model,optimizer,x,row_edges_index,col_edges_index)
        loss = train(model,optimizer,x,row_edges_index,col_edges_index)
        losses.append(loss)
        print(epoch,loss)
        z,cell_vectors = get_cell_vectors(model,x,row_edges_index,col_edges_index)
        vec_list=generate_table_vectors(cell_vectors,tokenized_tables,s_vocab=shuffled_vocab)
        result_score=evaluate_model(dataset,vec_list,k=5)
        print(result_score)
        results.append(result_score)

    


    # ### 3) Extract the latent cell vectors, generate table vectors:
    
    #z,cell_vectors = get_cell_vectors(model,x,train_pos_edge_index)





    #vec_list=generate_table_vectors(cell_vectors,tokenized_tables,s_vocab=shuffled_vocab)

    # ## 3) Evaluate the model
    #result_score=evaluate_model(dataset,vec_list,k=5)
    return cell_vectors,vec_list,losses,results

