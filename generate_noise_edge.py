import networkx as nx
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import torch
import random
import sys
dataset_str = 'cora'
dataset_str = ['cora','citeseer','pubmed']

def generate_noise_edges(dataset_str = 'cora', ratio):
    #for ratio in np.arange(10)/10.:
    with open('data/ind.{}.graph'.format(dataset_str), 'rb') as f:
        if sys.version_info > (3,0):
            graph_f = pkl.load(f, encoding='latin1')
        else:
            graph_f = pkl.load(f)
        G = nx.from_dict_of_lists(graph_f)
        edge_num = G.number_of_edges()
        node_num = G.number_of_nodes()
        noise_edge_num = int(edge_num * ratio)
        ii = 0 
        while(True):
            u,v = random.sample(range(node_num), 2)
            if not G.has_edge(u,v):
                G.add_edge(u,v)
                ii += 1
            if ii >= noise_edge_num:
                break
        print(G.number_of_edges())
        g_dict_of_list = nx.to_dict_of_lists(G)
        pkl.dump(g_dict_of_list, open('nsgcn/data/ind.{}.graph.noise.{}'.format(dataset_str, str(ratio)), 'wb'))


generate_noise_edges('cora', 0.1)
#print(G.edges())


    #adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_f))
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
        


