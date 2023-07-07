# Stabilised descent methods
![Platform: linux-64](https://img.shields.io/badge/linux--64-lightgray?label=platform&labelColor=gray&style=flat)
[![License: Hippocratic License HL3-CL-EXTR-MEDIA-SV](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-CL-EXTR-MEDIA-SV&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/cl-extr-media-sv.html)
[![arXiv Preprint](https://img.shields.io/badge/arXiv-2108.05237-b31b1b.svg)](https://arxiv.org/)

This repository contains the code for the paper [**Weighted sparse and low-rank least squares approximation**](https://arxiv.org/).
If you find this repository helpful for your work, please kindly cite this paper.
<pre>
@misc{trunschke2021convergence,
      title={Convergence bounds for nonlinear least squares and applications to tensor recovery},
      author={Philipp Trunschke},
      year={2021},
      eprint={2108.05237},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
</pre>

This repository is distributed under [hippocratic license](LICENSE.md).

## Installing dependencies
```
$ source setup install mamba
```

## Reproducing the experiments

```
$ source activate
(sparse_als)$ bash run_sampler.sh
(sparse_als)$ bash run_experiments.sh
```

## Running custom experiments

- The code currently only supports produc bases where every factor is the same.
- An example can be found in example.sh


# Old notes

## Sample generation
The samples that were used for the numerical experiment in the paper are provided in this repository.
To generate new samples, first create a directory (in the following referred to as `PROBLEM_DIRECTORY`) that contains the problem description file `parameters.json`.
Then execute the following commands to draw `10000` samples of the quantity of interst.
```
source activate
python compute_samples.py PROBLEM_DIRECTORY 10000
python compute_functional.py PROBLEM_DIRECTORY
```
For more details see `vmc_sampling/README.md`.

## Running experiments
The main procedure is located in `sparse_als.py`.
To run an approximation of the quantity of interest

$$
    U(y) := \int_{D} u(x,y) \,\mathrm{d}x,
$$

where $u$ is the solution of the uniform Darcy equation on the domain $D$, execute the following command.
```
source setup
python sparse_als.py -p uniform -t 100 -o 20 -d 15 -l corrected_gradient -r LInf
```

The tables from the paper can be reproduced by running `sparse_als.py` with the following set of parameters.

|       | Script |
|-------|--------|
|Table 1|`sparse_als.py -p uniform -t ...`|
|Table 2|`sparse_als.py -p lognormal -t ...`|
