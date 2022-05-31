import math
import os
import pickle
import re
import subprocess
import sys
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import pandas as pd
import warnings

from hook_interface import HookInterface

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.dirname(__file__)
print(MODEL_PATH)

start = time.time()


def time_since(since):
    now = time.time()
    s = now - since
    h = math.floor(s / 3600)
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '{}h {}min {:.2f} sec'.format(h, m, s)


class JITGNNHook(HookInterface):
    def __init__(self):
        super(JITGNNHook, self).__init__()

    def run_model(self):
        files = []
        for m in self.modifieds:
            command = subprocess.Popen(['git', 'show', 'HEAD:{}'.format(m)],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            output, error = command.communicate()
            if error.decode('utf-8'):
                print('error occurred by --- git show ---')
                sys.exit(-1)
            before = output.decode('utf-8')
            command = subprocess.Popen(['cat', '{}'.format(m)],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            output, error = command.communicate()
            if error.decode('utf-8'):
                print('error occurred by --- cat ---')
                sys.exit(-1)
            after = output.decode('utf-8')

            f_subtree = store_subtrees(m, before, after)
            if f_subtree:
                files.append(f_subtree)

        if all(f is None for f in files):
            print('nothing to evaluate')
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prd = main(files)
            print(prd)

        print(time_since(start))
        sys.exit(-1)


class GumTreeDiff:
    def __init__(self):
        self.bin_path = os.path.join(MODEL_PATH, 'gumtree-3.0.0/bin/gumtree')
        self.src_dir = os.path.join(MODEL_PATH, 'diff')
        if not os.path.exists(self.src_dir):
            os.makedirs(self.src_dir)

    def get_diff(self, fname, b_content, a_content):
        fname = fname.split('/')[-1]
        b_file = os.path.join(self.src_dir, fname.split('.')[0] + '_b.' + fname.split('.')[1])
        a_file = os.path.join(self.src_dir, fname.split('.')[0] + '_a.' + fname.split('.')[1])
        with open(b_file, 'w') as file:
            file.write(b_content)
        with open(a_file, 'w') as file:
            file.write(a_content)
        command = subprocess.Popen([self.bin_path, 'dotdiff', b_file, a_file],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        output, error = command.communicate()
        if error.decode('utf-8'):
            return None
        # remove files
        os.remove(b_file)
        os.remove(a_file)
        return output.decode('utf-8')

    def get_dotfiles(self, file):
        dot = self.get_diff(file[0], file[1], file[2])
        if dot is None:
            raise SyntaxError()
        lines = dot.splitlines()
        dotfiles = {
            'before': [],
            'after': []
        }
        current = 'before'
        node_pattern = '^n_[0-9]+_[0-9]+ \\[label=".+", color=(red|blue)\\];$'
        edge_pattern = '^n_[0-9]+_[0-9]+ -> n_[0-9]+_[0-9]+;$'

        for l in lines:
            if l == 'subgraph cluster_dst {':
                current = 'after'
            if not re.match(node_pattern, l) and not re.match(edge_pattern, l):
                continue
            dotfiles[current].append(l)

        return dotfiles['before'], dotfiles['after']


class SubTreeExtractor:
    def __init__(self, dot):
        self.dot = dot
        self.red_nodes = list()
        self.node_dict = dict()
        self.from_to = dict()  # mapping from src nodes to list of their dst nodes.
        self.to_from = dict()  # mapping from dst nodes to list of their src nodes.
        self.subtree_nodes = set()
        self.subtree_edges = set()

    def read_ast(self):
        node_pattern = '^n_[0-9]+_[0-9]+ \\[label=".+", color=(red|blue)\\];$'
        edge_pattern = '^n_[0-9]+_[0-9]+ -> n_[0-9]+_[0-9]+;$'
        for line in self.dot:
            if re.match(node_pattern, line):
                assert len(re.findall('^(.*) \\[label=".+", color=.+\\];$', line)) == 1
                id = re.findall('^(.*) \\[label=".+", color=.+\\];$', line)[0]
                assert len(re.findall('^n_[0-9]+_[0-9]+ \\[label="(.+)", color=.+\\];$', line)) == 1
                unclean_label = re.findall('^n_[0-9]+_[0-9]+ \\[label="(.+)", color=.+\\];$', line)[0]
                label = re.split('\\[[0-9]+', unclean_label)[0]
                assert len(re.findall('color=(red|blue)\\];$', line)) == 1
                color = re.findall('color=(red|blue)\\];$', line)[0]

                self.node_dict[id] = label
                if color == 'red':
                    self.red_nodes.append(id)

            elif re.match(edge_pattern, line):
                assert len(re.findall('^(.*) -> n_[0-9]+_[0-9]+;$', line)) == 1
                source = re.findall('^(.*) -> n_[0-9]+_[0-9]+;$', line)[0]
                assert len(re.findall('^n_[0-9]+_[0-9]+ -> (.*);$', line)) == 1
                dest = re.findall('^n_[0-9]+_[0-9]+ -> (.*);$', line)[0]

                if source not in self.from_to:
                    self.from_to[source] = [dest]
                else:
                    self.from_to[source].append(dest)
                if dest not in self.to_from:
                    self.to_from[dest] = [source]
                else:
                    self.to_from[dest].append(source)

            else:
                print(line, end='\t')

    def extract_subtree(self):
        self.read_ast()
        for n in self.red_nodes:
            self.subtree_nodes.add(n)
            if n in self.from_to:
                for d in self.from_to[n]:
                    self.subtree_nodes.add(d)
                    self.subtree_edges.add((n, d))
            if n in self.to_from:
                for s in self.to_from[n]:
                    self.subtree_nodes.add(s)
                    self.subtree_edges.add((s, n))
                    for d in self.from_to[s]:
                        self.subtree_nodes.add(d)
                        self.subtree_edges.add((s, d))

        vs = list(self.subtree_nodes)
        es = list(self.subtree_edges)
        colors = ['red' if node_id in self.red_nodes else "blue" for node_id in vs]
        features = [[self.node_dict[node_id]] if node_id in self.node_dict else ['unknown'] for node_id in vs]
        edges = [[], []]
        for src, dst in es:
            edges[0].append(vs.index(src))
            edges[1].append(vs.index(dst))

        return features, edges, colors

    def generate_dotfile(self):
        content = 'digraph G {\nnode [style=filled];\nsubgraph cluster_dst {\n'
        for node in self.subtree_nodes:
            content += '{} [label="{}", color={}];\n'.format(node,
                                                             self.node_dict[node],
                                                             'blue' if node not in self.red_nodes else 'red')
        for edge in self.subtree_edges:
            content += '{} -> {};\n'.format(edge[0], edge[1])
        content += '}\n;}\n'

        with open(os.path.join(MODEL_PATH, 'src', 'diff.dot'), 'w') as file:
            file.write(content)


class Dataset:
    def __init__(self, vectorizer, metrics_file, special_token):
        self.vectorizer_model = vectorizer
        self.special_token = special_token
        self.metrics = None
        self.load_metrics(metrics_file)
        # self.load_metrics(metrics_file)

    def load_metrics(self, metrics_file):
        self.metrics = pd.read_csv(os.path.join(MODEL_PATH, metrics_file))
        self.metrics = self.metrics.drop(
            ['author_date', 'bugcount', 'fixcount', 'revd', 'tcmt', 'oexp', 'orexp', 'osexp', 'osawr', 'project',
             'buggy', 'fix'],
            axis=1, errors='ignore')
        self.metrics = self.metrics[['commit_id', 'la', 'ld', 'nf', 'nd', 'ns', 'ent',
                                     'ndev', 'age', 'nuc', 'aexp', 'arexp', 'asexp']]
        self.metrics = self.metrics.fillna(value=0)

    @staticmethod
    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    @staticmethod
    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_adjacency_matrix(self, n_nodes, src, dst):
        edges = np.array([src, dst]).T
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(n_nodes, n_nodes),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # add supernode
        adj = sp.vstack([adj, np.ones((1, adj.shape[1]), dtype=np.float32)])
        adj = sp.hstack([adj, np.zeros((adj.shape[0], 1), dtype=np.float32)])
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        return adj

    def get_embedding(self, file_node_tokens, colors):
        for i, node_feat in enumerate(file_node_tokens):
            file_node_tokens[i] = node_feat.strip()
            if node_feat == 'N o n e':
                file_node_tokens[i] = 'None'
                colors.insert(i, 'blue')
                assert colors[i] == 'blue'
            if self.special_token:
                if ':' in node_feat:
                    feat_type = node_feat.split(':')[0]
                    file_node_tokens[i] = feat_type + ' ' + '<' + feat_type[
                                                                  :3].upper() + '>'  # e.g. number: 14 -> number <NUM>
        # fix the data later to remove the code above.
        features = self.vectorizer_model.transform(file_node_tokens).astype(np.float32)
        # add color feature at the end of features
        color_feat = [1 if c == 'red' else 0 for c in colors]
        features = sp.hstack([features, np.array(color_feat, dtype=np.float32).reshape(-1, 1)])
        # add supernode
        features = sp.hstack([features, np.zeros((features.shape[0], 1), dtype=np.float32)])
        supernode_feat = np.zeros((1, features.shape[1]), dtype=np.float32)
        supernode_feat[-1, -1] = 1
        features = sp.vstack([features, supernode_feat])
        features = self.normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
        return features

    def prepare_data(self, commit):
        c = ''
        try:
            metrics = torch.FloatTensor(self.normalize(self.metrics[self.metrics['commit_id'] == c]
                                                       .drop(columns=['commit_id']).to_numpy(dtype=np.float32))[0, :])
        except IndexError:
            # commit id not in commit metric set
            dim = self.metrics[self.metrics['commit_id'] == c].drop(columns=['commit_id']).shape[1]
            metrics = torch.zeros(dim, dtype=torch.float)

        b_node_tokens, b_edges, b_colors = [], [[], []], []
        a_node_tokens, a_edges, a_colors = [], [[], []], []
        b_nodes_so_far, a_nodes_so_far = 0, 0
        for file in commit:
            b_node_tokens += [' '.join(node) for node in file[1][0]]
            b_colors += [c for c in file[1][2]]
            b_edges = [
                b_edges[0] + [s + b_nodes_so_far for s in file[1][1][0]],   # source nodes
                b_edges[1] + [d + b_nodes_so_far for d in file[1][1][1]]    # destination nodes
            ]
            a_node_tokens += [' '.join(node) for node in file[2][0]]
            a_colors += [c for c in file[2][2]]
            a_edges = [
                a_edges[0] + [s + a_nodes_so_far for s in file[2][1][0]],   # source nodes
                a_edges[1] + [d + a_nodes_so_far for d in file[2][1][1]]    # destination nodes
            ]

            b_n_nodes = len(file[1][0])
            a_n_nodes = len(file[2][0])
            b_nodes_so_far += b_n_nodes
            a_nodes_so_far += a_n_nodes

        # if b_nodes_so_far + a_nodes_so_far > 28000 or b_nodes_so_far > 18000 or a_nodes_so_far > 18000:
        #     print('{} is a large commit, skip!'.format(c))
        #     return None

        before_embeddings = self.get_embedding(b_node_tokens, b_colors)
        before_adj = self.get_adjacency_matrix(b_nodes_so_far, b_edges[0], b_edges[1])
        after_embeddings = self.get_embedding(a_node_tokens, a_colors)
        after_adj = self.get_adjacency_matrix(a_nodes_so_far, a_edges[0], a_edges[1])
        training_data = [before_embeddings, before_adj, after_embeddings, after_adj, metrics]

        return training_data


class GraphConvolution(nn.Module):
    """
    from https://github.com/tkipf/pygcn/
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj):
        support = torch.mm(feature, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output += self.bias
        output = F.relu(output)
        output = F.dropout(output, p=0.2)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AttentionModule(torch.nn.Module):
    """
    Attention Module to make a pass on graph. from FuncGNN implementation at https://github.com/aravi11/funcGNN/
    """
    def __init__(self, size):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.size = size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.size,
                                                             self.size))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation


class TensorNetworkModule(torch.nn.Module):
    """
    funcGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, hidden_size, neuron_size):
        """
        :param args: Arguments object.
        """
        super(TensorNetworkModule, self).__init__()
        self.size = hidden_size
        self.tensor_neurons = neuron_size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.size,
                                                             self.size,
                                                             self.tensor_neurons))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.tensor_neurons,
                                                                   2*self.size))
        self.bias = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.mm(embedding_1.unsqueeze(0), self.weight_matrix.view(self.size, -1))
        scoring = scoring.view(self.size, self.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), torch.t(embedding_2.unsqueeze(0)))
        combined_representation = torch.cat((torch.t(embedding_1.unsqueeze(0)), torch.t(embedding_2.unsqueeze(0))))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        scores = torch.t(scores)
        return scores


class JITGNN(nn.Module):
    def __init__(self, hidden_size, message_size, metric_size):
        super(JITGNN, self).__init__()
        self.hidden_size = hidden_size
        self.message_size = message_size
        self.neuron_size = 32
        self.gnn11 = GraphConvolution(hidden_size, message_size)
        self.gnn12 = GraphConvolution(message_size, message_size)
        self.gnn13 = GraphConvolution(message_size, message_size)
        self.gnn14 = GraphConvolution(message_size, message_size)
        self.gnn21 = GraphConvolution(hidden_size, message_size)
        self.gnn22 = GraphConvolution(message_size, message_size)
        self.gnn23 = GraphConvolution(message_size, message_size)
        self.gnn24 = GraphConvolution(message_size, message_size)
        self.attention = AttentionModule(message_size)
        self.tensor_net = TensorNetworkModule(message_size, self.neuron_size)
        self.fc = nn.Linear(self.neuron_size + metric_size, 1)

    def forward(self, b_x, b_adj, a_x, a_adj, metrics):
        # change the design here. add adjacency matrix to graph convolution class so not pass it every time.
        b_node_embeddings = self.gnn14(self.gnn13(self.gnn12(self.gnn11(b_x, b_adj), b_adj), b_adj), b_adj)
        b_embedding = self.attention(b_node_embeddings[:-1, :]).flatten()
        a_node_embeddings = self.gnn24(self.gnn23(self.gnn22(self.gnn21(a_x, a_adj), a_adj), a_adj), a_adj)
        a_embedding = self.attention(a_node_embeddings[:-1, :]).flatten()
        # agg = torch.hstack([b_embedding, a_embedding])   # maybe a distance measure later
        agg = self.tensor_net(b_embedding, a_embedding).flatten()
        features = torch.hstack([agg, metrics])
        # features = agg
        output = self.fc(features)
        return output, agg


def predict(model, data):
    y_scores = []
    y_true = []
    # features_list = []
    # label_list = []
    model.eval()
    with torch.no_grad():
            if data is None:
                return -1
            model = model.to(device)
            output, features = model(data[0].to(device), data[1].to(device),
                                     data[2].to(device), data[3].to(device), data[4].to(device))
            # features_list.append(features)
            # label_list.append(label)
            prob = torch.sigmoid(output).item()
    return prob


def main(commit):
    with open(os.path.join(MODEL_PATH, 'vectorizer.pkl'), 'rb') as fp:
        vectorizer = pickle.load(fp)
    dataset = Dataset(vectorizer, 'apache_metrics_kamei.csv', special_token=False)
    data = dataset.prepare_data(commit)
    hidden_size = len(dataset.vectorizer_model.vocabulary_) + 2   # plus supernode node feature and node colors
    metric_size = dataset.metrics.shape[1] - 1
    message_size = 32
    model = JITGNN(hidden_size, message_size, metric_size)
    # model = torch.load(os.path.join(BASE_PATH, '34_model_best_auc.pt'), map_location=torch.device('cpu'))
    checkpoint = torch.load(os.path.join(MODEL_PATH, 'checkpoint.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    prob = predict(model, data)
    return prob


def store_subtrees(path, before, after):
    gumtree = GumTreeDiff()
    f = (path, before, after)
    try:
        b_dot, a_dot = gumtree.get_dotfiles(f)
    except SyntaxError:
        print('{} file is not supported.'.format(path))
        return None

    subtree = SubTreeExtractor(b_dot)
    b_subtree = subtree.extract_subtree()
    subtree = SubTreeExtractor(a_dot)
    a_subtree = subtree.extract_subtree()
    if len(b_subtree[0]) == 0 and len(a_subtree[0]) == 0:
        print('{} file is trivially changed.'.format(path))
        return None
    elif len(b_subtree[0]) == 0:
        b_subtree[0].append('None')
    elif len(a_subtree[0]) == 0:
        a_subtree[0].append('None')
    return path, b_subtree, a_subtree


# if __name__ == '__main__':
#     hook = JITGNNHook()
#     hook.run_model()

