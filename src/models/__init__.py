from .model import (
    bmshj2018_factorized,
    bmshj2018_factorized_relu,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    cheng2020_attn,
    mbt2018,
    mbt2018_mean,
    entroformer,
    datentroformer,
    datentroformerv1,
    datentroformerv2,
    entroformerdebug,
)

image_models = {
    "bmshj2018-factorized": bmshj2018_factorized,
    "bmshj2018-factorized-relu": bmshj2018_factorized_relu,
    "bmshj2018-hyperprior": bmshj2018_hyperprior,
    "mbt2018-mean": mbt2018_mean,
    "mbt2018": mbt2018,
    "cheng2020-anchor": cheng2020_anchor,
    "cheng2020-attn": cheng2020_attn,
    "entroformer":entroformer,
    "datentroformer":datentroformer,
    "datentroformerv1":datentroformerv1,
    "datentroformerv2":datentroformerv2,
    "entroformerdebug":entroformerdebug,

}