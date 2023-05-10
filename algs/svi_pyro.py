from torch import Tensor
from pyro import sample
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
            fn=Normal(z)
        )



svi = SVI(
    model,
    guide,
    optimizer,
    loss=Trace_ELBO()
)

