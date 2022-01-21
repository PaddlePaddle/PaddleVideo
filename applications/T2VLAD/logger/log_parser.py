import re
import scipy.stats
import logging
import numpy as np
from collections import defaultdict


def log_summary(logger, log_path, eval_mode="test_run", fixed_num_epochs=None):
    """Extract performace statistics from experiment log files.

    Args:
        logger (logger): reference to primary logging instance
        log_path (Path): the path to the log file
        eval_mode (str): the method use to collect the statistics. Can be one of:
            `test_run`, `fixed_num_epochs` or `geometric_mean`

    NOTE: The `eval_mode` argument differs by dataset: for datasets which provide a
    validation set, we use validation set performance to complete a single test run.  For
    datasets where no validation set is available, we aim to match prior work by either
    fixing the number of training epochs, or selecting directly from validation set
    performance (Details can be found in the supplementary material of the paper.)
    """
    with open(str(log_path), "r") as f:
        log = f.read().splitlines()

    # keep track of the random seed used for the part of the logfile being processed
    current_seed = None

    # Regex tag for finding the seed
    seed_tag = "Setting experiment random seed to"

    if eval_mode == "test_run":
        subset = "test"
    else:
        subset = "val"

    for mode in "t2v", "v2t":
        logger.info("")
        logger.info("----------------------------------------------------")
        logger.info(f"[{mode}] loaded log file with {len(log)} lines....")
        logger.info("----------------------------------------------------")

        # Search for the following metrics
        scores = {
            "R1": defaultdict(list),
            "R5": defaultdict(list),
            "R10": defaultdict(list),
            "R50": defaultdict(list),
            "MedR": defaultdict(list),
            "MeanR": defaultdict(list),
        }

        for row in log:
            if seed_tag in row:
                # Search for the log file entry describing the current random seed
                match = re.search(seed_tag + " (\d+)$", row)  # NOQA
                assert len(match.groups()) == 1, "expected a single regex match"
                current_seed = match.groups()[0]

            if f"{subset}_{mode}_metrics" in row:
                tokens = row.split(" ")
                for key in scores:
                    tag = f"{subset}_{mode}_metrics_{key}:"
                    if tag in tokens:
                        pos = tokens.index(tag) + 1
                        val = tokens[pos]
                        val = float(val)
                        assert current_seed is not None, "failed to determine the seed"
                        scores[key][current_seed].append(val)

        agg_scores = {"R1": [], "R5": [], "R10": [], "R50": [], "MedR": [], "MeanR": []}

        # compute the best performance for a single epoch (i.e. sharing the same model
        # to compute all stats)
        geometric_stats = defaultdict(list)
        best_epochs = {}
        if eval_mode == "geometric_mean":
            raise NotImplementedError("Need to fix this for new log format")
            consider = ["R1", "R5", "R10"]
            seeds = list(scores["R1"].keys())
            for seed in seeds:
                for metric, subdict in scores.items():
                    if metric in consider:
                        geometric_stats[seed].append(subdict[seed])
                gms_raw = np.array(geometric_stats[seed])
                geo_means = scipy.stats.mstats.gmean(gms_raw, axis=0)
                best_epochs[seed] = np.argmax(geo_means)

        for metric, subdict in scores.items():
            for seed, values in subdict.items():
                if eval_mode == "test_run":
                    stat = values[0]
                elif eval_mode == "fixed_num_epochs":
                    stat = values[fixed_num_epochs - 1]
                elif "LSMDC" in log_path and eval_mode == "geometric_mean":
                    stat = values[best_epochs[seed]]
                else:
                    raise ValueError(f"unrecognised eval_mode: {eval_mode}")
                agg_scores[metric].append(stat)

        if eval_mode == "fixed_num_epochs":
            logger.info(f"Reporting stats with fixed training length: {fixed_num_epochs}")
        for metric, values in agg_scores.items():
            logger.info(f"{metric}: {np.mean(values):.1f}, {np.std(values, ddof=1):.1f}")
