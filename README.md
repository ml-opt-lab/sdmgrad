# sdmgrad

This repo contains codes for SDMGrad.

## Toy Example
To run the toy example:
```
python toy.py
```

## Consistency Verification
The experiments are conducted on the multi-Fashion+MNIST dataset, which can be downloaded from [ParetoMTL](https://github.com/Xi-L/ParetoMTL). Then follow the `run.sh` script to run the experiments.

## Supervised Learning
The expriments are conducted on Cityscapes and NYU-v2 datasets, which can be downloaded from [MTAN](https://github.com/lorenmt/mtan). Then follow the `run.sh` script to run the experiments and `evaluate.py` to evaluate the performance. 

## Reinforcement Learning
The experiments are conducted on [Meta-World](https://github.com/Farama-Foundation/Metaworld) benchmark. To run the experiments on `MT10` and `MT50` (the instructions below are partly borrowed from [CAGrad](https://github.com/Cranial-XIX/CAGrad)):

1. Create python3.6 virtual environment. The `requirements.txt` file is attached in the `mtrl` folder.
2. Install the [MTRL](https://github.com/facebookresearch/mtrl) codebase.
3. Install the [Meta-World](https://github.com/Farama-Foundation/Metaworld) environment with commit id `d9a75c451a15b0ba39d8b7a8b6d18d883b8655d8`.
4. Copy the `mtrl_files` folder to the `mtrl` folder in the installed mtrl repo, then 

```
cd PATH_TO_MTRL/mtrl_files/ && chmod +x mv.sh && ./mv.sh
```

5. Follow the `run.sh` to run the experiments.
