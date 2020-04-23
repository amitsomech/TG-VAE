###############################
####        Plots        ######
###############################

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils.vectorize import get_vector

def plot_compound_loss(losses,limit=-1):
    ls=[]
    rrls=[]
    crls=[]
    rkls=[]
    ckls=[]
    for l,rrl,crl,rkl,ckl in losses[:limit]:
        ls.append(float(l))
        rrls.append(float(rrl))
        crls.append(float(crl))
        rkls.append(float(rkl))
        ckls.append(float(ckl))
        
    data = pd.DataFrame({"rrl":rrls,"crl":crls,"rkl":rkls,"ckl":ckls,"step":range(len(rrls))})
    
    cols=[]
    for c in ['rrl','crl','rkl','ckl']:        
        cols.append(data[c])
    x=data.step.values
    y=cols
    plt.stackplot(x,y, labels=['Rows Recon loss','Column Recon-loss','Row KL-Loss','Column KL-Loss'])
    plt.plot(data.step,np.sum(y,axis=0),"black",label="Overall Loss") 
    #plt.plot(data.step,data.rkl,label="Row KL-Loss",linewidth=2)
    #plt.plot(data.step,data.ckl,label="Column KL-Loss",linewidth=2)
    plt.xlabel('Epoch Num.')
    plt.ylabel('Loss')
    plt.legend()
    

def plot_sample_vectors(cell_vectors,sample_size=1000):
    idx = np.random.randint(len(cell_vectors), size=sample_size)
    sample = cell_vectors[idx,:]
    vals = TSNE(n_components=2,init='pca').fit_transform(sample)
    sns.scatterplot(x=vals[:,0],y=vals[:,1])


def plot_single_table_vectors(table,s_vocab=None):
    vec_table=[]
    for col_id in range(table.shape[1]):
        #print(col_id)
        vec_table+=[{"vector":get_vector(x,cell_vectors,s_vocab=s_vocab),"col_id":col_id} for _,x in np.ndenumerate(table[:,col_id])]

    vec_df=pd.DataFrame(vec_table)
    x = TSNE(n_components=2,init='pca').fit_transform(list(vec_df.vector.values))
    vec_df["x"]=x.T[0]
    vec_df["y"]=x.T[1]
    vec_df.col_id=vec_df.col_id.astype("category")
    sns.scatterplot(x="x",y="y",data=vec_df,hue="col_id")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def plot_single_table_relative_vectors(table,cell_vectors,s_vocab=None,labels=None,sample=1.0):
    vec_table=[]
    for col_id in range(table.shape[1]):
        #print(col_id)
        vec_table+=[{"vector":get_vector(x,cell_vectors,s_vocab=s_vocab),"col_id":col_id} for _,x in np.ndenumerate(table[:,col_id])]
    mean_vec =np.nanmean([v["vector"] for v in vec_table],axis=0)
    #print(mean_vec)
    vec_df=pd.DataFrame(vec_table)
    vec_list = list(vec_df.vector.values)
    vec_list.append(mean_vec)
    #print(vec_list)
    tsne_vecs = TSNE(n_components=2,init='pca').fit_transform(vec_list)
    x = tsne_vecs[:-1]
    mean_tsne= tsne_vecs[-1]
    vec_df["x"]=x.T[0]
    vec_df["y"]=x.T[1]
    vec_df.col_id=vec_df.col_id.astype("category")
    vec_df["str_vec"]=vec_df.vector.apply(str)
    g=pd.DataFrame(vec_df.groupby("str_vec").vector.count())
    nv=vec_df[["x","y","str_vec","col_id"]].drop_duplicates("str_vec").set_index("str_vec")
    final=nv.join(g)
    final=final.sample(frac=sample)
    if labels is not None:
        final["col_id"]= final["col_id"].apply(lambda x:labels[x])
    clrs = sns.color_palette('tab20', n_colors=len(final.col_id.unique()))
    sns.relplot(x="x", y="y", hue="col_id", size="vector",
                sizes=(40,400), alpha=.8,
                height=6, data=final,palette=clrs)
    plt.scatter(mean_tsne[0],mean_tsne[1],marker='X',s=100,color='red')

def plot_table_vectors(vec_list,table_dataset):
    vals = TSNE(n_components=2,init='pca').fit_transform(vec_list)
    d=pd.DataFrame(vals)
    labels=[(t["id"],t["label"]) for t in table_dataset]
    dd=pd.DataFrame(labels)
    dd.columns=["tid","label"]
    d.columns=["x","y"]
    d=d.join(dd["label"])
    sns.scatterplot(data=d,x="x",y="y",hue="label")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
