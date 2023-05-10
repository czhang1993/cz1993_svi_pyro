from torch import Tensor
from pyro import param, sample
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


def model(data):
    mu = Tensor(0.0)
    sigma = Tensor(1.0)
    z = sample(
        name="z",
        fn=Normal(mu, sigma)
    )
    for i in range(len(data)):
        sample(
            name="obs_{}".format(i),
            fn=Normal(z),
            obs=data[i]
        )
        
        
def guide(data):
    q_mu = param(
        name="q_mu",
        init_tensor=Tensor(0.0)
    )
    q_sigma = param(
        name="q_sigma",
        init_tensor=Tensor(1.0)
    )
    sample(
        name="z",
        fn=Normal(q_mu, q_sigma)
    )
    
    
optim = Adam()
    
svi = SVI(
    model=model,
    guide=guide,
    optim=optim,
    loss=Trace_ELBO()
)
