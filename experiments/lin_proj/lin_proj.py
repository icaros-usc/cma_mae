
import os
import csv
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

from dask.distributed import Client, LocalCluster

from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import (AnnealingEmitter, GaussianEmitter, 
                           IsoLineEmitter, ImprovementEmitter,
                           OptimizingEmitter,
                           GradientEmitter, GradientAnnealingEmitter,
                           GradientImprovementEmitter)
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap

def calc_sphere(sol):
    
    dim = sol.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    target_shift = 5.12 * 0.4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - target_shift)**2 * dim
    raw_obj = np.sum(np.square(sol - target_shift), axis=1)
    objs = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    derivatives = -2 * (sol - target_shift)

    return objs, derivatives

def calc_rastrigin(sol):

    A = 10.0
    dim = sol.shape[1]

    # Shift the Rastrigin function so that the optimal value is at x_i = 2.048.
    target_shift = 5.12 * 0.4

    best_obj = np.zeros(len(sol))
    displacement = -5.12 * np.ones(sol.shape) - target_shift
    sum_terms = np.square(displacement) - A * np.cos(2 * np.pi * displacement)
    worst_obj = 10 * dim + np.sum(sum_terms, axis=1)

    displacement = sol - target_shift
    sum_terms = np.square(displacement) - A * np.cos(2 * np.pi * displacement)
    raw_obj = 10 * dim + np.sum(sum_terms, axis=1)

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    # Approximate 0 by the bottom-left corner.
    objs = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    derivatives = -(2 * displacement + 2 * np.pi * A * np.sin(2 * np.pi * displacement))

    return objs, derivatives

def calc_plateau(sol):
    n = sol.shape[1]
    clipped = sol.copy()
    clip_indices = np.where(np.logical_or(clipped > 5.12, clipped < -5.12))

    # All objective outputs are the same. Output 100.
    raw_obj = np.zeros(sol.shape)
    # Apply a quadratic penalty for any component outside the range [-5.12, 5.12].
    raw_obj[clip_indices] = np.square(np.abs(clipped[clip_indices])-5.12)
    raw_obj = np.average(raw_obj, axis=1)

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = np.zeros(len(sol))
    worst_obj = np.full(len(sol), 100.0)
    objs = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # The derivatives are all 0 for the flat portion.
    derivatives = np.zeros(sol.shape)
    # The derivative is -2 for the quadratic portions.
    derivatives[clip_indices] = -2.0

    return objs, derivatives

# Batch calculate the lin projection for all solutions given.
def calc_measures(sol):

    dim = sol.shape[1]

    # Calculate BCs.
    clipped = sol.copy()
    clip_indices = np.where(np.logical_or(clipped > 5.12, clipped < -5.12))
    clipped[clip_indices] = 5.12 / clipped[clip_indices]
    measures = np.concatenate(
        (
            np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
        ),
        axis=1,
    )

    derivatives = np.ones(sol.shape)
    derivatives[clip_indices] = -5.12 / np.square(sol[clip_indices])
    
    mask_0 = np.concatenate((np.ones(dim//2), np.zeros(dim-dim//2)))
    mask_1 = np.concatenate((np.zeros(dim//2), np.ones(dim-dim//2)))

    d_measure0 = np.multiply(derivatives, mask_0)
    d_measure1 = np.multiply(derivatives, mask_1)
    
    jacobian = np.stack((d_measure0, d_measure1), axis=1)
 
    return measures, jacobian


def create_optimizer(algorithm, dim, alpha=1.0, resolution=100, seed=None):
    """Creates an optimizer based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        dim (int): Dimensionality of the sphere function.
        alpha (float): The archive learning rate.
        resolution (int): The archive resolution (res x res).
        seed (int): Main seed or the various components.
    Returns:
        Optimizer: A ribs Optimizer for running the algorithm.
    """
    max_bound = dim / 2 * 5.12
    bounds = [(-max_bound, max_bound), (-max_bound, max_bound)]
    initial_sol = np.zeros(dim)
    batch_size = 36
    num_emitters = 15
    grid_dims = (resolution, resolution)
    
    # Create archive.
    if algorithm in [
            "map_elites", "map_elites_line",
            "cma_me", "cma_me_star", "cma_mega",
            "cma_me_io",
    ]:
        archive = GridArchive(grid_dims, bounds, seed=seed)
    elif algorithm in ["cma_mae", "cma_maega"]:
        archive = GridArchive(
                grid_dims, bounds, 
                archive_learning_rate=alpha, 
                seed=seed,
        )
    else:
        raise ValueError(f"Algorithm `{algorithm}` is not recognized")

    # Maintain a passive elitist archive
    passive_archive = GridArchive(grid_dims, bounds, seed=seed)
    passive_archive.initialize(dim)

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if seed is None else list(
        range(seed, seed + num_emitters))
    if algorithm in ["map_elites"]:
        emitters = [
            GaussianEmitter(archive,
                            initial_sol,
                            0.5,
                            batch_size=batch_size,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["map_elites_line"]:
        emitters = [
            IsoLineEmitter(archive,
                           initial_sol,
                           iso_sigma=0.5,
                           line_sigma=0.2,
                           batch_size=batch_size,
                           seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_mega"]:
        emitters = [
            GradientImprovementEmitter(archive,
                            initial_sol,
                            sigma_g=10.0,
                            stepsize=1.0,
                            gradient_optimizer="gradient_ascent",
                            normalize_gradients=True,
                            restart_rule='basic',
                            selection_rule="mu",
                            bounds=None,
                            batch_size=batch_size - 1,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_maega"]:
        emitters = [
            GradientAnnealingEmitter(archive,
                    initial_sol,
                    sigma_g=10.0,
                    stepsize=1.0,
                    gradient_optimizer="gradient_ascent",
                    normalize_gradients=True,
                    restart_rule='basic',
                    bounds=None,
                    batch_size=batch_size - 1,
                    seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me"]:
        emitters = [
            ImprovementEmitter(archive,
                               initial_sol,
                               0.5,
                               restart_rule='basic',
                               selection_rule='mu',
                               batch_size=batch_size,
                               seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me_io"]:
        emitters = []
        split_count = len(emitter_seeds) // 2
        emitters += [
            OptimizingEmitter(archive,
                              initial_sol,
                              0.5,
                              restart_rule='basic',
                              selection_rule='mu',
                              batch_size=batch_size,
                              seed=s) for s in emitter_seeds[:split_count]
        ]
        emitters += [
            ImprovementEmitter(archive,
                               initial_sol,
                               0.5,
                               restart_rule='basic',
                               selection_rule='mu',
                               batch_size=batch_size,
                               seed=s) for s in emitter_seeds[split_count:]
        ]
    elif algorithm in ["cma_me_star"]:
        emitters = [
            ImprovementEmitter(archive,
                               initial_sol,
                               0.5,
                               restart_rule='no_improvement',
                               selection_rule='filter',
                               batch_size=batch_size,
                               seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_mae"]:
        emitters = [
            AnnealingEmitter(archive,
                             initial_sol,
                             0.5,
                             restart_rule='basic',
                             batch_size=batch_size,
                             seed=s) for s in emitter_seeds
        ]

    return Optimizer(archive, emitters), passive_archive

def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=100, cmap='viridis')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())

def run_experiment(algorithm,
                   trial_id,
                   total_trials,
                   dim=100,
                   alpha=1.0,
                   arch_res_exp=False,
                   resolution=100,
                   objective='sphere',
                   init_pop=100,
                   itrs=10000,
                   outdir="logs",
                   log_freq=1,
                   log_arch_freq=1000,
                   seed=None):
    
    # Create a directory for this specific trial.
    if algorithm in ["cma_mae", "cma_maega"]:
        s_logdir = os.path.join(outdir, f"{algorithm}_{resolution}_{alpha}", f"trial_{trial_id}")
    else:
        s_logdir = os.path.join(outdir, f"{algorithm}_{resolution}", f"trial_{trial_id}")
    logdir = Path(s_logdir)
    if not logdir.is_dir():
        logdir.mkdir()
    
    # Create a new summary file
    summary_filename = os.path.join(s_logdir, f"summary.csv")
    if os.path.exists(summary_filename):
        os.remove(summary_filename)
    with open(summary_filename, 'w') as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(['Iteration', 'QD-Score', 'Coverage', 'Maximum', 'Average'])
   
    # If we are running a resolution experiment, override the resolution.
    if arch_res_exp:

        # Linearly interpolate
        index = 0.0
        if total_trials > 1: 
            index = trial_id / (total_trials-1.0)

        min_count = 50
        max_count = 500
        new_resolution = index * (max_count - min_count) + min_count
        resolution = int(new_resolution + 1e-9)
        cell_count = resolution ** 2

        ratio = cell_count / (100.0 * 100.0)
        alpha = 1.0 - np.power(1.0 - alpha, ratio)

        print('Running exp {} with resolution = {} and alpha = {}'.format(
            trial_id, resolution, alpha))

    is_init_pop = algorithm in [
        'map_elites', 'map_elites_line',
    ]
    is_dqd = algorithm in [
        'cma_mega', 'cma_maega',
    ]

    # Select the objective based on the input.
    obj_func = None
    if objective == 'plateau':
        obj_func = calc_plateau
    elif objective == 'sphere':
        obj_func = calc_sphere
    elif objective == 'Rastrigin':
        obj_func = calc_rastrigin

    optimizer, passive_archive = create_optimizer(
            algorithm, dim,
            alpha=alpha, 
            resolution=resolution,
            seed=seed,
    )
    archive = optimizer.archive

    best = 0.0
    non_logging_time = 0.0
    with alive_bar(itrs) as progress:

        if is_init_pop:
            # Sample initial population
            sols = np.array([np.random.normal(size=dim) for _ in range(init_pop)])

            objs, _ = obj_func(sols)
            best = max(best, max(objs))
            measures, _ = calc_measures(sols)

            # Add each solution to the archive.
            for i in range(len(sols)):
                archive.add(sols[i], objs[i], measures[i])
                passive_archive.add(sols[i], objs[i], measures[i])

        for itr in range(1, itrs + 1):
            itr_start = time.time()

            if is_dqd:
                sols = optimizer.ask(grad_estimate=True)
                objs, jacobian_obj = obj_func(sols)
                best = max(best, max(objs))
                measures, jacobian_measure = calc_measures(sols)
                jacobian_obj = np.expand_dims(jacobian_obj, axis=1)
                jacobian = np.concatenate((jacobian_obj, jacobian_measure), axis=1)
                optimizer.tell(objs, measures, jacobian=jacobian)

                # Update the passive elitist archive.
                for i in range(len(sols)):
                    passive_archive.add(sols[i], objs[i], measures[i])

            sols = optimizer.ask()
            objs, _ = obj_func(sols)
            best = max(best, max(objs))
            measures, _ = calc_measures(sols)
            optimizer.tell(objs, measures)

            # Update the passive elitist archive.
            for i in range(len(sols)):
                passive_archive.add(sols[i], objs[i], measures[i])

            non_logging_time += time.time() - itr_start
            progress()
            
            # Save the archive at the given frequency.
            # Always save on the final iteration.
            final_itr = itr == itrs
            if (itr > 0 and itr % log_arch_freq == 0) or final_itr:

                # Save a full archive for analysis.
                df = passive_archive.as_pandas(include_solutions = final_itr)
                df.to_pickle(os.path.join(s_logdir, f"archive_{itr:08d}.pkl"))

                # Save a heatmap image to observe how the trial is doing.
                save_heatmap(passive_archive, os.path.join(s_logdir, f"heatmap_{itr:08d}.png"))

            # Update the summary statistics for the archive
            if (itr > 0 and itr % log_freq == 0) or final_itr:
                with open(summary_filename, 'a') as summary_file:
                    writer = csv.writer(summary_file)
                    
                    sum_obj = 0
                    num_filled = 0
                    num_bins = passive_archive.bins
                    for sol, obj, beh, idx, meta in zip(*passive_archive.data()):
                        num_filled += 1
                        sum_obj += obj    
                    qd_score = sum_obj / num_bins
                    average = sum_obj / num_filled
                    coverage = 100.0 * num_filled / num_bins
                    data = [itr, qd_score, coverage, best, average]
                    writer.writerow(data)


def lin_proj_main(algorithm,
                  trials=20,
                  parallel=True,
                  arch_res_exp=False,
                  dim=100,
                  alpha=1.0,
                  resolution=100,
                  objective='sphere',
                  init_pop=100,
                  itrs=10000,
                  outdir="logs",
                  log_freq=1,
                  log_arch_freq=1000,
                  seed=None):
    """Experiment tool for the lin_proj domain from the CMA-ME paper.

    Args:
        algorithm (str): Name of the algorithm.
        trials (int): Number of experimental trials to run.
        parallel (bool): Should each trial be run in parallel or serial?
        arch_res_exp (bool): Runs the archive resolution experiment instead.
        dim (int): Dimensionality of solutions.
        alpha (float): The archive learning rate.
        resolution (int): The resolution of dimension in the archive (res x res).
        objective (str): Either sphere or Rastrigin as the objective. By default, use sphere.
        init_pop (int): Initial population size for MAP-Elites (ignored for CMA variants).
        itrs (int): Iterations to run.
        outdir (str): Directory to save output.
        log_freq (int): Number of iterations between computing QD metrics and updating logs.
        log_arch_freq (int): Number of iterations between saving an archive and generating heatmaps.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """
   
    if objective not in ["plateau", "sphere", "Rastrigin"]:
        raise ValueError(f"Objective `{objective}` is not recognized.")
 
    if arch_res_exp:
        print(f"Running arch res experiment on {objective} at n={dim} and alpha={alpha}.")
    else:
        print(f"Running {objective} at n={dim}, alpha={alpha}, and resolution={resolution}.")

    # Create a shared logging directory for the experiments for this algorithm.
    if algorithm in ["cma_mae", "cma_maega"]:
        s_logdir = os.path.join(outdir, f"{algorithm}_{resolution}_{alpha}")
    else:
        s_logdir = os.path.join(outdir, f"{algorithm}_{resolution}")
    logdir = Path(s_logdir)
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()
    if not logdir.is_dir():
        logdir.mkdir()
 
    exp_func = lambda cur_id: run_experiment(
            algorithm, cur_id,
            total_trials=trials,
            dim=dim,
            alpha=alpha,
            arch_res_exp=arch_res_exp,
            resolution=resolution,
            objective=objective,
            init_pop=init_pop,
            itrs=itrs,
            outdir=outdir,
            log_freq=log_freq,
            log_arch_freq=log_arch_freq,
            seed=seed,
        )

    # Run all trials in parallel.
    if parallel:
        cluster = LocalCluster(
            processes=True,  # Each worker is a process.
            n_workers=trials,  # Create one worker per trial (assumes >=trials cores)
            threads_per_worker=1,  # Each worker process is single-threaded.
        )
        client = Client(cluster)
        # Run an experiment as a separate process to run all exps in parallel.
        trial_ids = list(range(trials))
        futures = client.map(exp_func, trial_ids)
        results = client.gather(futures)

    # Run all trials sequentially.
    else:
        for cur_id in range(trials):
            exp_func(cur_id)

if __name__ == '__main__':
    fire.Fire(lin_proj_main)
