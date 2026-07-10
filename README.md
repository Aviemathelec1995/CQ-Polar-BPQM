# CQ Polar BPQM

Simulation code for the paper [Polar Codes for CQ Channels: Decoding via Belief-Propagation with Quantum Messages](https://arxiv.org/abs/2401.07167).

This repository implements density-evolution design tools and a classical simulation of a paired-measurement belief-propagation with quantum messages (PM-BPQM) successive-cancellation decoder for polar codes over binary-input symmetric classical-quantum (BSCQ) qubit channels.

## Table of Contents

* Background
* Repository Layout
* Requirements
* Installation
* Examples
* Usage
* Generated Outputs
* Notes
* Citation
* License

## Background

The paper studies polar-code construction and decoding for classical-quantum channels using paired-measurement BPQM. For a qubit BSCQ channel, the channel can be represented by parameters `delta` and `gamma` through

```text
rho(delta, gamma) = [[delta, gamma],
                     [gamma, 1 - delta]]
W(x) = sigma_x^x rho(delta, gamma) sigma_x^x.
```

The simulations in this repository focus on:

* Density evolution for qubit BSCQ channel parameters.
* Check-node and bit-node PM-BPQM combining operations.
* Polar-code information-set selection from synthesized-channel error estimates.
* Classical Monte Carlo simulation of the quantum successive-cancellation PM-BPQM decoder.
* Block-error and bit-error estimates for short polar codes.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `BSCQ_SCD_Polar_Decoder.py` | Main script. It implements BSCQ state construction, PM-BPQM check-node and bit-node updates, polar density evolution, polar encoding, recursive CQ polar decoding, and Monte Carlo error-rate estimation. |
| `BSCQ_SCD_Parameter_Sweep.py` | Parameter-sweep runner that reuses the main decoder, evaluates one or more `(delta, gamma)` channel settings, and writes descriptive JSON output for density-evolution and optional Monte Carlo decoder results. |
| `BSCQ_SCD_Polar_Decoder.ipynb` | Notebook version of the BSCQ successive-cancellation polar decoder workflow. |
| `polar_cq_decoder_random_codeword.ipynb` | Notebook for CQ polar-decoder simulations using randomly sampled codewords. |

## Requirements

* Python 3.10 or newer.
* Scientific Python packages:
  * `numpy`
  * `scipy`
  * `numba`
  * `matplotlib`

Some systems may need compiler tools for `numba` and scientific Python wheels.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone <repo-url>
cd CQ-Polar-BPQM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy numba matplotlib
```

Run commands from the repository root. If you run scripts from another directory, add the repository root to `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/CQ-Polar-BPQM:${PYTHONPATH}"
```

## Examples

Run a short length-4 simulation:

```bash
python BSCQ_SCD_Polar_Decoder.py -n 2 -M 100 -nde 100 -d 0.05 -g 0.15 -s 123
```

Run a length-8 simulation:

```bash
python BSCQ_SCD_Polar_Decoder.py -n 3 -M 1000 -nde 20000 -d 0.05 -g 0.15 -s 123
```

Run a length-16 simulation:

```bash
python BSCQ_SCD_Polar_Decoder.py -n 4 -M 1000 -nde 20000 -d 0.07 -g 0.10 -s 123
```

Run a density-evolution-only parameter sweep and save JSON results:

```bash
python BSCQ_SCD_Parameter_Sweep.py -n 3 -nde 1000 --density-only --delta-values 0.03,0.05,0.07 --gamma-values 0.10,0.15 -s 123
```

Run the sweep with decoder Monte Carlo samples:

```bash
python BSCQ_SCD_Parameter_Sweep.py -n 3 -M 100 -nde 1000 --delta-values 0.05,0.07 --gamma-values 0.10,0.15 -s 123
```

## Usage

The main script accepts:

```text
-d      BSCQ delta parameter
-g      BSCQ gamma parameter
-n      number of polar stages, so block length is 2^n
-M      number of Monte Carlo decoder samples
-nde    number of density-evolution samples
-s      random seed
```

For example:

```bash
python BSCQ_SCD_Polar_Decoder.py -d 0.05 -g 0.15 -n 3 -M 500 -nde 5000 -s 7
```

The script currently has built-in information sets for block lengths `N = 4`, `N = 8`, and `N = 16`.

The parameter-sweep script accepts the same `-d`, `-g`, `-n`, `-M`, `-nde`, and `-s` arguments. It also accepts:

```text
--delta-values    comma-separated delta values for a sweep
--gamma-values    comma-separated gamma values for a sweep
--density-only    run density evolution without Monte Carlo decoder samples
--output-dir      directory for JSON output
--output-file     JSON output file name
```

## Generated Outputs

The command-line script prints:

* Density-evolution error-rate estimates for each synthesized polar channel.
* Simulated channel error rates.
* Simulated block error rate.
* `BER_NUM` and `BER_DEN`, which can be used to estimate first-error-location rates.

The script does not write output files by default.

`BSCQ_SCD_Parameter_Sweep.py` writes a JSON payload under `results/` by default. Each result records the channel parameters, block length, seed, information set, density-evolution error estimates, and, unless `--density-only` is used, simulated channel-error, block-error, `BER_NUM`, and `BER_DEN` values.

## Notes

* The local checkout does not include a `polar_bpqm.pdf` file. The implementation was checked against the arXiv paper linked above.
* The decoder is a classical simulation of PM-BPQM operations. Direct classical simulation scales exponentially in block length, so practical runs are limited to short codes.
* The BSCQ convention in the paper has Helstrom error rate `delta` for small `delta`. A recommended decoder correction is to use the `|1>` projection as the decision-0 outcome in the base-case hard decision.
* The current command-line seed path creates a `RandomState` object but does not pass it into the simulation routines. For reproducible command-line runs, seed the global NumPy RNG or thread an explicit RNG through the sampling functions.
* Large values of `M` and `nde` can be slow because the simulation repeatedly applies quantum-state tensor operations and Numba-compiled density-evolution updates.

## Citation

If you use this code, cite:

```bibtex
@article{mandal2024polarcqbpqm,
  title={Polar Codes for CQ Channels: Decoding via Belief-Propagation with Quantum Messages},
  author={Mandal, Avijit and Brandsen, Sam and Pfister, Henry D.},
  journal={arXiv preprint arXiv:2401.07167},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.
