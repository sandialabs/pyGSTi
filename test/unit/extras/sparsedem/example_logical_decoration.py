"""
End-to-end example: learn a sparse DEM and decorate it with logical flags.

Pipeline:
  1. Construct a ground-truth graph-like DEM: 20 events on 10 detectors
     (10 single-detector "boundary" events + 10 detector-pair events forming
     a ring), each with a small probability, decorated with logical flags.
     The flags form a graph cut (detectors {0..4} vs {5..9} + boundary), so
     they are cycle-consistent, as for a real code's logical observable.
  2. Simulate shots by sampling events independently; XOR detector patterns
     to get syndromes and XOR flags of fired events to get the true logical
     outcome y per shot.
  3. Learn an UN-decorated DEM from the syndrome data alone with the lattice
     pruning algorithm (pygsti.extras.sparsedem).
  4. Apply the logical decoration tool: build a pymatching matcher from the
     LEARNED DEM, decode every shot to get the event indicator matrix B,
     solve Y = B L over GF(2) for the flags, and decorate the learned DEM.
  5. Report learned-vs-true events, recovered-vs-true flags, the residual
     (inconsistent-row) fraction, and rank/determinacy diagnostics.

Run with:
    python example_logical_decoration.py [--shots N] [--seed S]
        [--decoder pymatching|tesseract]

The decoder used in step 4 is selectable: pymatching (minimum-weight perfect
matching; graph-like DEMs only) or tesseract (most-likely-error decoding;
also handles hyperedge events with more than two detectors).
"""

import argparse
import time

import numpy as np

from pygsti.extras.sparsedem.core import SparseDEMEstimator
from pygsti.extras.sparsedem.io import dem_to_dict
from pygsti.extras.sparsedem.logical_decoration import assign_logical_flags
from pygsti.extras.sparsedem.utils import counts_from_samples

N_DETECTORS = 10


def mask_to_str(mask):
    return " ".join(f"D{d}" for d in range(N_DETECTORS) if (mask >> d) & 1)


def build_ground_truth():
    """20 distinct graph-like events with probabilities and logical flags."""
    masks, probs, flags = [], [], []
    # 10 single-detector (boundary) events. Cut = {D0..D4} vs {D5..D9}+boundary,
    # so boundary events on D0..D4 flip the logical.
    for d in range(N_DETECTORS):
        masks.append(1 << d)
        probs.append(0.008 + 0.001 * d)
        flags.append(1 if d < 5 else 0)
    # 10 detector-pair events forming a ring 0-1-...-9-0. Ring edges (4,5) and
    # (9,0) cross the cut.
    for d in range(N_DETECTORS):
        d2 = (d + 1) % N_DETECTORS
        masks.append((1 << d) | (1 << d2))
        probs.append(0.010 + 0.001 * d)
        flags.append(1 if d in (4, 9) else 0)
    order = np.argsort(masks)
    return (
        np.array(masks)[order],
        np.array(probs)[order],
        np.array(flags, dtype=np.uint8)[order],
    )


def simulate(masks, probs, flags, n_shots, rng):
    """Sample events independently; XOR patterns and flags per shot."""
    H = np.zeros((N_DETECTORS, len(masks)), dtype=np.uint8)
    for j, mask in enumerate(masks):
        for d in range(N_DETECTORS):
            H[d, j] = (int(mask) >> d) & 1
    occurrences = (rng.random((n_shots, len(masks))) < probs).astype(np.uint8)
    syndromes = ((occurrences @ H.T) % 2).astype(np.uint8)
    y = ((occurrences @ flags.astype(np.int64)) % 2).astype(np.uint8)
    return syndromes, y


def main(n_shots=4000, seed=2026, decoder="pymatching"):
    rng = np.random.default_rng(seed)
    true_masks, true_probs, true_flags = build_ground_truth()
    true_flag_of = dict(zip(true_masks.tolist(), true_flags.tolist()))
    true_prob_of = dict(zip(true_masks.tolist(), true_probs.tolist()))

    print(f"Ground truth: {len(true_masks)} events on {N_DETECTORS} detectors")
    print(f"Shots: {n_shots}")
    if n_shots > 1500:
        print(f"  (note: using {n_shots} shots; ~1000 shots gives too little "
              "signal for the lattice pruning z-tests at these event rates)")

    syndromes, y = simulate(true_masks, true_probs, true_flags, n_shots, rng)
    print(f"Observed logical-flip rate: {y.mean():.4f}")

    # --- Learn the UN-decorated DEM from detector data only -----------------
    counts = dict(counts_from_samples(syndromes))
    estimator = SparseDEMEstimator(counts)
    learned_dem = estimator.estimate_lattice_pruned(confidence=0.999)
    learned = dem_to_dict(learned_dem)
    learned_masks = np.array(sorted(learned.keys()))

    found = [m for m in true_masks if m in learned]
    missed = [m for m in true_masks if m not in learned]
    spurious = [m for m in learned_masks if m not in set(true_masks.tolist())]
    print("\n--- Learned DEM vs ground truth ---")
    print(f"Learned events: {len(learned_masks)}  "
          f"(found {len(found)}/{len(true_masks)} true, "
          f"{len(spurious)} spurious, {len(missed)} missed)")
    for m in missed:
        print(f"  missed:   {mask_to_str(m):18s} (p_true = {true_prob_of[m]:.4f})")
    for m in spurious:
        print(f"  spurious: {mask_to_str(m):18s} (p_learned = {learned[m]:.4f})")
    prob_errs = [abs(learned[m] - true_prob_of[m]) for m in found]
    print(f"Max |p_learned - p_true| over found events: {max(prob_errs):.4f}")

    # --- Assign logical flags to the learned DEM ----------------------------
    t0 = time.perf_counter()
    decorated_dem, flags, residual, diag = assign_logical_flags(
        learned_dem, syndromes, y, num_detectors=N_DETECTORS,
        decoder=decoder, seed=seed
    )
    elapsed = time.perf_counter() - t0

    print("\n--- Logical flag assignment ---")
    print(f"Decoder backend: {decoder}  "
          f"(decode + GF(2) solve took {elapsed:.2f} s for {n_shots} shots)")
    print(f"Residual (inconsistent-row) fraction: {residual:.5f} "
          f"({int(round(residual * n_shots))} of {n_shots} shots)")
    print(f"GF(2) solve method: {diag['method']} "
          f"(initial residual {diag['initial_residual_fraction']:.5f}, "
          f"{diag['ransac_iterations']} RANSAC iterations)")
    print(f"rank(B) = {diag['rank']} of {diag['n_events']} decoded events; "
          f"converged = {diag['converged']}")
    undetermined = set(int(m) for m in diag["undetermined_masks"])
    if undetermined:
        print(f"Undetermined flags for events: "
              f"{[mask_to_str(m) for m in sorted(undetermined)]}")
    else:
        print("All event flags are determined by the data.")

    flag_of = dict(zip(diag["masks"].tolist(), flags.tolist()))
    n_correct = 0
    n_checked = 0
    print(f"\n{'event':18s} {'p_true':>7s} {'p_learn':>8s} "
          f"{'flag_true':>9s} {'flag_rec':>8s}")
    for m in sorted(learned_masks.tolist()):
        rec = flag_of[m]
        if m in true_flag_of:
            truth = true_flag_of[m]
            ok = "" if rec == truth else "  <-- MISMATCH"
            if m not in undetermined:
                n_checked += 1
                n_correct += int(rec == truth)
            print(f"{mask_to_str(m):18s} {true_prob_of[m]:7.4f} "
                  f"{learned[m]:8.4f} {truth:9d} {rec:8d}{ok}")
        else:
            note = " (undetermined)" if m in undetermined else ""
            print(f"{mask_to_str(m):18s} {'--':>7s} {learned[m]:8.4f} "
                  f"{'--':>9s} {rec:8d}  [spurious]{note}")

    print(f"\nFlag recovery on correctly-learned, determined events: "
          f"{n_correct}/{n_checked}")
    print(f"Decorated DEM has {decorated_dem.num_observables} observable(s) "
          f"and {len(dem_to_dict(decorated_dem))} events.")

    print("\nRESULT:", "SUCCESS" if n_correct == n_checked else "FLAG MISMATCH")
    return 0 if n_correct == n_checked else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--decoder", choices=["pymatching", "tesseract"],
                        default="pymatching")
    args = parser.parse_args()
    raise SystemExit(main(args.shots, args.seed, args.decoder))
