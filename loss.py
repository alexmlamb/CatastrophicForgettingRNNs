#!/usr/bin/env python

import sys
sys.path.append("/u/lambalex/DeepLearning/dreamprop/lib")

import theano
import theano.tensor as T

class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()

def cast(inp):
    return T.cast(inp, 'float32')

def crossent(p,y): 
    return -T.mean(T.log(p)[T.arange(y.shape[0]), y])

def nll(p,y,n):
    return -T.sum(T.log(p)*expand(y,n))

def accuracy(p,y):
    return T.mean(cast(T.eq(cast(T.argmax(p, axis = 1)),cast(y))))

def expand(y,n):
    return T.extra_ops.to_one_hot(y, n)

def clip_updates(updates, params):
    new_updates = []
    for (p,u) in updates.items():
        if p in params:
            print "UPDATING P IN PARAMS"
            u = T.clip(u, -0.01, 0.01)
        new_updates.append((p,u))
    return new_updates

def lsgan_loss(D_q_lst, D_p_lst, bs=True):
    dloss = 0.0
    gloss = 0.0

    max_len = max(len(D_q_lst), len(D_p_lst))

    for i in range(len(D_p_lst)):
        D_p = D_p_lst[i]
        dloss += T.mean(T.sqr(0.0 - D_p))
        gloss += T.mean(T.sqr(1.0 - D_p))

    for i in range(len(D_q_lst)):
        D_q = D_q_lst[i]
        dloss += T.mean(T.sqr(1.0 - D_q))
        gloss += T.mean(T.sqr(0.0 - D_q))

    return dloss / max_len, gloss / max_len

def improvement_loss(D1lst, D2lst): 
    new_loss = 0.0

    for i in range(len(D1lst)):
        D1 = D1lst[i]
        D2 = D2lst[i]
        print "loss - just push up D2"
        new_loss += T.mean(T.switch(T.lt(D2,D1)*T.lt(D2,0.9),-1.0*D2,0.0))
        new_loss += T.mean(T.switch(T.gt(D2,1.0), 1.0*D2, 0.0))

    return new_loss


def wgan_loss(D_q_lst, D_p_lst):
    dloss = 0.0
    gloss = 0.0

    #for dloss, push up D_p and push down D_q
    #for gloss, push up D_q and push down D_p

    for i in range(len(D_q_lst)):
        D_q = D_q_lst[i]
        D_p = D_p_lst[i]
        dloss += T.mean(D_q) + T.mean(-1.0 * D_p)
        gloss += T.mean(D_p) + T.mean(-1.0 * D_q)

    return dloss / len(D_q_lst), gloss / len(D_q_lst)
    


if __name__ == "__main__":

    p = T.matrix()
    y = T.ivector()

    f = theano.function([p,y], [crossent(p,y), nll(p,y)], allow_input_downcast=True)

    print f([[0.9,0.1],[0.1,0.9]], [0,0])
    print f([[0.9,0.1]], [1])
    print f([[0.7,0.3]], [0])
    print f([[0.5,0.5]], [0])
    print f([[0.1,0.9]], [0])


