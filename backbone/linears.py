'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import logging
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
# from timm.models.layers.weight_init import trunc_normal_.  # timm > 1.0 in timm.layers....
from torch.nn.init import trunc_normal_

class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, nb_proxy=1, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = (out_features1 + out_features2) * nb_proxy
        self.nb_proxy = nb_proxy
        self.fc1 = CosineLinear(in_features, out_features1, nb_proxy, False, False)
        self.fc2 = CosineLinear(in_features, out_features2, nb_proxy, False, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1['logits'], out2['logits']), dim=1)  # concatenate along the channel

        # Reduce_proxy
        out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {
            'old_scores': reduce_proxies(out1['logits'], self.nb_proxy),
            'new_scores': reduce_proxies(out2['logits'], self.nb_proxy),
            'logits': out
        }


class EaseCosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(EaseCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)
    
    def reset_parameters_to_zero(self):
        self.weight.data.fill_(0)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}
    
    def forward_reweight(self, input, cur_task, alpha=0.1, beta=0.0, init_cls=10, inc=10, out_dim=768, use_init_ptm=False):
        for i in range(cur_task + 1):
            if i == 0:
                start_cls = 0
                end_cls = init_cls
            else:
                start_cls = init_cls + (i - 1) * inc
                end_cls = start_cls + inc
            
            out = 0.0
            for j in range((self.in_features // out_dim)):
                # PTM feature
                if use_init_ptm and j == 0:
                    input_ptm = F.normalize(input[:, 0:out_dim], p=2, dim=1)
                    weight_ptm = F.normalize(self.weight[start_cls:end_cls, 0:out_dim], p=2, dim=1)
                    out_ptm = beta * F.linear(input_ptm, weight_ptm)
                    out += out_ptm
                    continue

                input1 = F.normalize(input[:, j*out_dim:(j+1)*out_dim], p=2, dim=1)
                weight1 = F.normalize(self.weight[start_cls:end_cls, j*out_dim:(j+1)*out_dim], p=2, dim=1)
                if use_init_ptm:
                    if j != (i+1):
                        out1 = alpha * F.linear(input1, weight1)
                        out1 /= cur_task
                    else:
                        out1 = F.linear(input1, weight1)
                else:
                    if j != i:
                        out1 = alpha * F.linear(input1, weight1)
                        out1 /= cur_task
                    else:
                        out1 = F.linear(input1, weight1)

                out += out1
            
            if i == 0:
                out_all = out
            else:
                out_all = torch.cat((out_all, out), dim=1) if i != 0 else out
                
        if self.to_reduce:
            # Reduce_proxy
            out_all = reduce_proxies(out_all, self.nb_proxy)

        if self.sigma is not None:
            out_all = self.sigma * out_all
        
        return {'logits': out_all}


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm
        heads = []
        single_head = []
        if with_norm:
            single_head.append(nn.LayerNorm(embed_dim))

        single_head.append(nn.Linear(embed_dim, nb_classes, bias=True))
        head = nn.Sequential(*single_head)

        heads.append(head)
        self.heads = nn.ModuleList(heads)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 

    def backup(self):
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        self.load_state_dict(self.old_state_dict)

    def update(self, nb_classes, freeze_old=True):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))
            
        _fc = nn.Linear(self.embed_dim, nb_classes, bias=True)
        trunc_normal_(_fc.weight, std=.02)
        nn.init.constant_(_fc.bias, 0) 
        single_head.append(_fc)
        new_head = nn.Sequential(*single_head)

        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad=False

        self.heads.append(new_head)

    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(self.heads[ti](fc_inp))
        out = {'logits': torch.cat(out, dim=1)}
        return out
    
class LabelFeatureCounter:
    def __init__(self):
        self.feature_counts = {}

    def update_counts(self, labels):
        for lbl in labels.unique():
            lbl = lbl.item()
            if lbl in self.feature_counts:
                self.feature_counts[lbl] += (labels == lbl).sum().item()
            else:
                self.feature_counts[lbl] = (labels == lbl).sum().item()

    def get_count(self, label):
        return self.feature_counts.get(label.item(), 0)


class NearestNeighborClassifier(nn.Module):
    def __init__(self, similarity_metric='cosine'):
        super(NearestNeighborClassifier, self).__init__()
        self.support_features = None
        self.support_labels = None
        # self.register_parameter('support_features', None)
        # self.register_parameter('support_labels', None)
        self.similarity_metric = similarity_metric  # 'cosine' or 'l2'
        self.label_feature_counter = LabelFeatureCounter()
        self.debug_verbose = 5

    @property
    def ready(self):
        return self.support_features is not None and self.support_labels is not None
    
    def set_support(self, features, labels):
        """
        Set or update the support set for the nearest neighbor classifier.
        Args:
            features: Tensor of shape (num_support_samples, feature_dim)
            labels: Tensor of shape (num_support_samples,)
        """
        features = F.normalize(features, p=2, dim=1) if self.similarity_metric == 'cosine' else features
        unique_labels = labels.unique()
        
        if self.debug_verbose > 0:
            logging.debug(f"Setting support features and labels: {unique_labels}")

        if self.support_features is None or self.support_labels is None:
            # Initialize support features and labels
            self.support_features = torch.stack([features[labels == lbl].mean(dim=0) for lbl in unique_labels])
            self.support_labels = unique_labels
            self.label_feature_counter.update_counts(labels)
        else:
            # Update support features by averaging with counts
            for lbl in unique_labels:
                new_feature = features[labels == lbl].mean(dim=0)
                new_count = (labels == lbl).sum().item()

                if lbl in self.support_labels:
                    idx = (self.support_labels == lbl).nonzero(as_tuple=True)[0].item()
                    old_count = self.label_feature_counter.get_count(lbl)
                    total_count = old_count + new_count
                    self.support_features[idx] = (self.support_features[idx] * old_count + new_feature * new_count) / total_count
                    self.label_feature_counter.feature_counts[lbl.item()] = total_count
                else:
                    self.support_features = torch.cat((self.support_features, new_feature.unsqueeze(0)), dim=0)
                    self.support_labels = torch.cat((self.support_labels, lbl.unsqueeze(0)), dim=0)
                    self.label_feature_counter.feature_counts[lbl.item()] = new_count
                    
        if self.debug_verbose > 0:
            logging.debug(f"Updated support features and labels: {self.support_labels}")
            logging.debug(f"Support features: {self.support_features.shape} {self.support_features[0, :5]}")
            logging.debug(f"Counter: {self.label_feature_counter.feature_counts}")
            
            self.debug_verbose -= 1
        

    def forward(self, query_features):
        """
        Perform nearest neighbor classification.
        Args:
            query_features: Tensor of shape (num_query_samples, feature_dim)
        Returns:
            A dictionary with 'logits' containing the similarity scores.
        """
        if self.support_features is None or self.support_labels is None:
            # raise ValueError("Support set not initialized. Call set_support() first.")
            return {'logits': query_features[:, :100]}     # dummy return

        if self.similarity_metric == 'cosine':
            query_features = F.normalize(query_features, p=2, dim=1)
            similarities = torch.mm(query_features, self.support_features.t())  # Cosine similarity
        elif self.similarity_metric == 'l2':
            distances = torch.cdist(query_features, self.support_features)  # L2 distances
            similarities = -distances  # Convert distances to similarities

        return {'logits': similarities}
    