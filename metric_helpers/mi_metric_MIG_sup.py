import torch

metric_name = 'MIG'


def MIG(mi_normed):
    return torch.mean(mi_normed[:, 0] - mi_normed[:, 1])


def compute_metric_shapes(marginal_entropies, cond_entropies, active_units):
    factor_entropies = [6, 40, 32, 32]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    mutual_infos_s1 = torch.sort(mi_normed, dim=1, descending=True)[0].clamp(min=0)    
    metric = eval('MIG')(mutual_infos_s1)
    mutual_infos_s2 = torch.sort(mi_normed.transpose(0,1), dim=1, descending=True)[0].clamp(min=0)
    metric_sup = eval('MIG')(mutual_infos_s2[active_units,:])
    print("MIG", metric, "MIG-sup", metric_sup)
    return metric, metric_sup 


def compute_metric_faces(marginal_entropies, cond_entropies):
    factor_entropies = [21, 11, 11]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[0].clamp(min=0)
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    metric = eval(metric_name)(mi_normed)
    return metric


def compute_metric_3dshapes(marginal_entropies, cond_entropies, active_units):
    factor_entropies = [10, 10, 10, 8, 4, 15]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    mutual_infos_s1 = torch.sort(mi_normed, dim=1, descending=True)[0].clamp(min=0)
    metric = eval('MIG')(mutual_infos_s1)
    mutual_infos_s2 = torch.sort(mi_normed.transpose(0, 1), dim=1, descending=True)[0].clamp(min=0)
    metric_sup = eval('MIG')(mutual_infos_s2[active_units,:])
    print('MIG', metric, 'MIG-sup', metric_sup)
    return metric, metric_sup

def compute_metric_mpi3D(marginal_entropies, cond_entropies, active_units):
    factor_entropies = [4,4,2,3,3,40,40]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    mutual_infos_s1 = torch.sort(mi_normed, dim=1, descending=True)[0].clamp(min=0)
    metric = eval('MIG')(mutual_infos_s1)
    mutual_infos_s2 = torch.sort(mi_normed.transpose(0, 1), dim=1, descending=True)[0].clamp(min=0)
    metric_sup = eval('MIG')(mutual_infos_s2[active_units,:])
    print('MIG', metric, 'MIG-sup', metric_sup)
    return metric, metric_sup
    

