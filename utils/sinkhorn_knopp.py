import torch


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def iterate(self, Q):#Q(C,N) r->c
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q #归一化
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]#r class number #假设先验条件：类别概率均等
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]#c samples number #先验条件：样本概率
        for it in range(self.num_iters):#iterate:i(row),j(col) Q(i)*r(i)/Q(i).sum(),Q(j)*c(j)/Q(j).sum()
            u = torch.sum(Q, dim=1)#(p for each class)
            u = r / u# (C)
            u = shoot_infs(u)#
            Q *= u.unsqueeze(1)#乘以每类的概率
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)#乘以每个样本概率
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @torch.no_grad()
    def forward(self, logits):
        # get assignments
        q = logits / self.epsilon #logits shape()#sharpen
        M = torch.max(q)
        q -= M
        q = torch.exp(q).t()#convert to Probability
        return self.iterate(q)

def optimize_L_sk(PS, nh=0):
    import numpy as np
    import time
    # PS
    #N = max(self.L.size())#样本数

    tt = time.time()
    #self.PS = self.PS.T  #
    PS=PS.T#now it is K x N
    PS/=np.sum(PS)
    N=PS.shape[1]
    r = np.ones((PS.shape[0], 1), dtype=PS.dtype) / PS.shape[0]  # (K,1)
    c = np.ones((N, 1), dtype=PS.dtype) / N  # (N,1)
    #self.PS **= self.lamb  # K x N
    inv_K = 1.0/PS.shape[0]#self.dtype(1. / self.outs[nh]) #self.outs[nh]=K
    inv_N = 1.0/N
    err = 1e6
    _counter = 0
    while err > 1e-2:
        r = inv_K / (PS @ c)  # (KxN)@(N,1) = K x 1
        c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
        #if _counter % 10 == 0:
        err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    print("error: ", err, 'step ', _counter, flush=True)  # " nonneg: ", sum(I), flush=True)
    # inplace calculations.
    PS *= np.squeeze(c)
    PS = PS.T
    PS *= np.squeeze(r)
    #PS = PS.T
    PS=(PS.T / np.sum(PS.T, axis=0, keepdims=True)).T

    argmaxes = np.nanargmax(PS, 0)  # size N
    newL = torch.LongTensor(argmaxes)
    # self.L[nh] = newL.to(self.dev)
    # newL.cuda()
    print('opt took {0:.2f}min, {1:4d}iters'.format(((time.time() - tt) / 60.), _counter), flush=True)
    return PS,_counter


if __name__=="__main__":

#1
    # Q=torch.diag(torch.ones([5]),diagonal=0).cuda()#1
#2
    # Q=torch.ones([5,6]).cuda()#2
#3
    # Q = torch.rand((5, 6), dtype=torch.float32, device='cuda')#3
    # ind=torch.rand([5,1],dtype=torch.float32)
    # Q=torch.repeat_interleave(ind,6,dim=1).cuda()#(N,C)
    Q=torch.rand((4,5)).cuda()
    Q=torch.diag(torch.tensor([1,1,1]),0).cuda().float()
    #Q = Q / self.epsilon #logits shape()#sharpen
    M = torch.max(Q)
    Q-= M
    Q = torch.exp(Q)
    Q1,c=optimize_L_sk(Q.cpu().numpy())
    sk=SinkhornKnopp(3,1.0)
    Qo=sk(Q)
    print(Q1)
    print(Qo)