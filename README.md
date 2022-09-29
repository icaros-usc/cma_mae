# Covariance Matrix Adaptation MAP-Annealing

This repository contains code for the experiments of the Covariance Matrix Adaptation MAP-Annealing (CMA-MAE) paper.

The project contains a modified version of [pyribs](https://pyribs.org), a quality diversity optimization library, from the [Differentiable Quality Diversity](https://proceedings.neurips.cc/paper/2021/hash/532923f11ac97d3e7cb0130315b067dc-Abstract.html) paper ([github](https://github.com/icaros-usc/dqd)). We implement the CMA-MAE and CMA-MAEGA algorithms in pyribs. The `AnnealingEmitter` (see `ribs/emitters/_annealing_emitter.py`) implements the CMA-MAE algorithm and the `GradientAnnealingEmitter` (see `ribs/emitters/_gradient_annealing_emitter.py`) implements the CMA-MAEGA algorithm. We modify `ArchiveBase` (see `ribs/archives/_archive_base.py`) to implement acceptance thresholds needed by both algorithms.

## Requirements

The project builds in [Anaconda](www.anaconda.com).

Once installed, create the conda environment with the following command:

```bash
conda env create -f experiments/environment.yml

```

Next activate the conda environment and install pyribs:

```bash
conda activate cma_mae_exps
pip3 install -e .[all]
```

## Pretrained Models

To run LSI (StyleGAN) experiments, you must first download the StyleGAN pretrained models from the StyleGAN [repo](https://github.com/lernapparat/lernapparat/releases/download/v2019-02-01/karras2019stylegan-ffhq-1024x1024.for_g_all.pt). Place the `.pt` file in the folder `experiments/lsi_clip`.

To run LSI (StyleGAN2) experimenets, you must first download the StyleGAN2 pretrained model from the Nvidia [website](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2/files). Place the `.pt` file in the folder `experiments/lsi_clip_2/models`.

CLIP automatically installs with the conda environment.

## StyleGAN2 Additional Code

We include dnnlib and torch_util from the StyleGAN2-Ada [repo](https://github.com/NVlabs/stylegan2-ada) in `experiments/lsi_clip_2` for replicability.

## Running Experiments

For each experiment you pick an identifier for the algorithm you want to run.

| Quality Diversity Algorithm | Identifier         |
| --------------------------- | ------------------:|
| MAP-Elites                  | map_elites         |
| MAP-Elites (line)           | map_elites_line    |
| CMA-ME                      | cma_me             |
| CMA-ME (imp, opt)           | cma_me_io          |
| CMA-ME*                     | cma_me_star        |
| CMA-MAE                     | cma_mae            |
| CMA-MEGA                    | cma_mega           |
| CMA-MAEGA                   | cma_maega          |

### Linear Projection (sphere)

To run an experiment with MAP-Elites:

```bash
conda activate cma_mae_exps
cd experiments/lin_proj

python3 lin_proj.py map_elites --objective sphere
```

To run a different algorithm replace `map_elites` with another identifier from the above table.

For additional options see:

```bash
python3 lin_proj.py --help

```

### Linear Projection (Rastrigin)


To run an experiment with MAP-Elites:

```bash
conda activate cma_mae_exps
cd experiments/lin_proj

python3 lin_proj.py map_elites --objective Rastrigin
```

To run a different algorithm replace `map_elites` with another identifier from the above table.

For additional options see:

```bash
python3 lin_proj.py --help

```

### Linear Projection (plateau)


To run an experiment with MAP-Elites:

```bash
conda activate cma_mae_exps
cd experiments/lin_proj

python3 lin_proj.py map_elites --objective plateau
```

To run a different algorithm replace `map_elites` with another identifier from the above table.

For additional options see:

```bash
python3 lin_proj.py --help

```

### Arm Repertoire

To run an experiment with MAP-Elites:

```bash
conda activate cma_mae_exps
cd experiments/arm

python3 arm.py map_elites
```

To run a different algorithm replace `map_elites` with another identifier from the above table.

For additional options see:

```bash
python3 arm.py --help

```

### Latent Space Illumination (StyleGAN)

To run an experiment with MAP-Elites:

```bash
conda activate cma_mae_exps
cd experiments/lsi_clip

python3 lsi.py map_elites 
```

To run a different algorithm replace `map_elites` with another identifier from the above table.

For additional options see:

```bash
python3 lsi.py --help

```

### Latent Space Illumination (StyleGAN2)

To run an experiment with MAP-Elites:

```bash
conda activate cma_mae_exps
cd experiments/lsi_clip_2

python3 lsi.py map_elites 
```

To run a different algorithm replace `map_elites` with another identifier from the above table.

For additional options see:

```bash
python3 lsi.py --help

```

## Results

The following tables contain the reported results from the paper and commands to run each experiment.

### Linear Projection (sphere)

| Quality Diversity Algorithms  | QD-score    | Coverage   | Experiment Command                                            |
| ----------------------------  | ----------: | ---------: | :------------------------------------------------------------ |
| MAP-Elites                    | 41.64       | 50.80%     | `python3 lin_proj.py map_elites --objective sphere`           |
| MAP-Elites (line)             | 49.07       | 60.42%     | `python3 lin_proj.py map_elites_line --objective sphere`      |
| CMA-ME                        | 36.50       | 42.82%     | `python3 lin_proj.py cma_me --objective sphere`               |
| CMA-MAE                       | 64.86       | 83.31%     | `python3 lin_proj.py cma_mae --alpha 0.01 --objective sphere` |

### Linear Projection (Rastrigin)

| Quality Diversity Algorithms  | QD-score    | Coverage   | Experiment Command                                               |
| ----------------------------  | ----------: | ---------: | :--------------------------------------------------------------- |
| MAP-Elites                    |  31.43      | 47.88%     | `python3 lin_proj.py map_elites --objective Rastrigin`           |
| MAP-Elites (line)             |  38.29      | 56.51%     | `python3 lin_proj.py map_elites_line --objective Rastrigin`      |
| CMA-ME                        |  38.02      | 53.09%     | `python3 lin_proj.py cma_me --objective Rastrigin`               |
| CMA-MAE                       |  52.65      | 80.46%     | `python3 lin_proj.py cma_mae --alpha 0.01 --objective Rastrigin` |

### Linear Projection (plateau)

| Quality Diversity Algorithms  | QD-score    | Coverage   | Experiment Command                                             |
| ----------------------------  | ----------: | ---------: | :------------------------------------------------------------- |
| MAP-Elites                    |  47.07      | 47.07%     | `python3 lin_proj.py map_elites --objective plateau`           |
| MAP-Elites (line)             |  52.20      | 52.20%     | `python3 lin_proj.py map_elites_line --objective plateau`      |
| CMA-ME                        |  34.54      | 34.54%     | `python3 lin_proj.py cma_me --objective plateau`               |
| CMA-MAE                       |  79.27      | 79.29%     | `python3 lin_proj.py cma_mae --alpha 0.01 --objective plateau` |

### Arm Repertoire 

| Quality Diversity Algorithms  | QD-score    | Coverage   | Experiment Command                    |
| ----------------------------  | ----------: | ---------: | :------------------------------------ |
| MAP-Elites                    | 71.40       | 74.09%     | `python3 arm.py map_elites`           |
| MAP-Elites (line)             | 74.55       | 75.61%     | `python3 arm.py map_elites_line`      |
| CMA-ME                        | 75.82       | 75.89%     | `python3 arm.py cma_me`               |
| CMA-MAE                       | 79.03       | 79.24%     | `python3 arm.py cma_mae --alpha 0.01` |

### Latent Space Illumination (StyleGAN)

| Quality Diversity Algorithms  | QD-score    | Coverage   | Experiment Command                   |
| ----------------------------  | ----------: | ---------: | :----------------------------------- |
| MAP-Elites                    | 12.85       | 19.42%     | `python3 lsi.py map_elites`          |
| MAP-Elites (line)             | 14.40       | 21.11%     | `python3 lsi.py map_elites_line`     |
| CMA-ME                        | 14.00       | 19.57%     | `python3 lsi.py cma_me`              |
| CMA-MAE                       | 17.67       | 25.08%     | `python3 lsi.py cma_mae --alpha 0.1` |

### Latent Space Illumination (StyleGAN2)

| Quality Diversity Algorithms  | QD-score    | Coverage   | Experiment Command                   |
| ----------------------------  | ----------: | ---------: | :----------------------------------- |
| MAP-Elites                    | -276.18     | 4.48%      | `python3 lsi.py map_elites`          |
| MAP-Elites (line)             | -827.25     | 8.81%      | `python3 lsi.py map_elites_line`     |
| CMA-MEGA                      | 9.18        | 14.91%     | `python3 lsi.py cma_mega`            |
| CMA-MAEGA                     | 11.51       | 18.62%     | `python3 lsi.py cma_maega`           |

See the paper and supplementary materials for full data and standard error bars.

## License

pyribs and this project are both released under the MIT License.

[pyribs MIT License](https://github.com/icaros-usc/pyribs/blob/master/LICENSE)
