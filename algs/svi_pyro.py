from torch import Tensor
from pyro import param, sample
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO


def model(data):
    mu = Tensor(0.0)
    sigma = Tensor(1.0)
    z = sample(
        name="z",
        fn=Normal(mu, sigma)
    )
    for i in range(n):
        sample(
            name=?,
            fn=Normal(z),
            ?
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
        name="q_z",
        fn=Normal(q_mu, q_sigma)
    )
    

svi = SVI(
    model,
    guide,
    optimizer,
    loss=Trace_ELBO()
)
