import argparse as ap
import json
import time
from pathlib import Path

import numpy as np

from BSCQ_SCD_Polar_Decoder import (
  helstrom_success,
  pauli,
  polar,
  polar_decoder_cq_sample_output_random_frozen_avg_error,
  rhom,
)


def parse_float_list(raw_value):
  '''
    Parse a comma-separated list of floating-point values.

          Arguments:
                  raw_value(str): comma-separated values such as "0.03,0.05"

          Returns:
                  values(float[:]): parsed values as a list of floats
  '''
  return [float(value.strip()) for value in raw_value.split(',') if value.strip()]


def builtin_info_set(number_qubits):
  '''
    Return the built-in polar information set used by the reference decoder.

          Arguments:
                  number_qubits(int): polar block length

          Returns:
                  info_set(int[:]): 1 indicates information bit, 0 indicates frozen bit
                  num_frozen(int): number of frozen bits
  '''
  if number_qubits==4:
    return [0,0,1,1], 2
  if number_qubits==8:
    return [0,0,0,1,0,1,1,1], 4
  if number_qubits==16:
    return [0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1], 8
  raise ValueError('built-in information sets are available only for N=4, N=8, and N=16')


def density_evolution_error_rates(delta,gamma,n,num_de_samples):
  '''
    Compute synthesized-channel error estimates using density evolution.

          Arguments:
                  delta(float): BSCQ delta parameter
                  gamma(float): BSCQ gamma parameter
                  n(int): number of polar stages
                  num_de_samples(int): number of density-evolution samples

          Returns:
                  error_rates(float[:]): Helstrom error estimates for synthesized channels
  '''
  d=np.ones(num_de_samples)*delta
  g=np.ones(num_de_samples)*gamma
  synthesized_channels=polar(n,d,g)
  error_rates=[]
  for i in range(2**n):
    channel_delta=np.mean(synthesized_channels[i][0])
    channel_gamma=np.mean(synthesized_channels[i][1])
    rho0=rhom(channel_delta,channel_gamma,0)
    error_rates.append(float(1-helstrom_success(rho0,pauli(1))))
  return error_rates


def run_single_experiment(delta,gamma,n,M,num_de_samples,seed,density_only=False):
  '''
    Run one BSCQ polar-code experiment.

          Arguments:
                  delta(float): BSCQ delta parameter
                  gamma(float): BSCQ gamma parameter
                  n(int): number of polar stages
                  M(int): number of Monte Carlo decoder samples
                  num_de_samples(int): number of density-evolution samples
                  seed(int): NumPy random seed
                  density_only(bool): skip Monte Carlo decoder simulation when true

          Returns:
                  result(dict): JSON-serializable experiment result
  '''
  number_qubits=2**n
  info_set,num_frozen=builtin_info_set(number_qubits)

  np.random.seed(seed)
  density_errors=density_evolution_error_rates(delta,gamma,n,num_de_samples)

  result={
    'delta': float(delta),
    'gamma': float(gamma),
    'n': int(n),
    'number_qubits': int(number_qubits),
    'num_de_samples': int(num_de_samples),
    'num_decoder_samples': int(M),
    'seed': int(seed),
    'info_set': [int(value) for value in info_set],
    'num_frozen': int(num_frozen),
    'density_evolution_error_rate': density_errors,
  }

  if density_only:
    result['decoder_simulation_skipped'] = True
    return result

  channel_error,block_error,ber_num,ber_den=polar_decoder_cq_sample_output_random_frozen_avg_error(
    delta,
    gamma,
    M,
    number_qubits,
    info_set,
    num_frozen,
  )

  result.update({
    'simulated_channel_error_rate': channel_error.tolist(),
    'block_error_rate': float(block_error),
    'ber_num': ber_num.tolist(),
    'ber_den': ber_den.tolist(),
  })
  return result


def summarize_result(result):
  '''
    Build a compact text summary for one experiment result.

          Arguments:
                  result(dict): result returned by run_single_experiment

          Returns:
                  summary(str): printable one-line summary
  '''
  summary=(
    f"d={result['delta']}, g={result['gamma']}, N={result['number_qubits']}, "
    f"min_DE={min(result['density_evolution_error_rate']):.6g}, "
    f"max_DE={max(result['density_evolution_error_rate']):.6g}"
  )
  if 'block_error_rate' in result:
    summary=summary+f", block_error={result['block_error_rate']:.6g}"
  return summary


def write_results(results,output_dir,filename):
  '''
    Write sweep results to a JSON file.

          Arguments:
                  results(dict): full sweep result payload
                  output_dir(str): directory where the output file is written
                  filename(str): JSON file name

          Returns:
                  output_path(Path): path to the written JSON file
  '''
  output_path=Path(output_dir)
  output_path.mkdir(parents=True,exist_ok=True)
  result_file=output_path/filename
  with result_file.open('w',encoding='utf-8') as handle:
    json.dump(results,handle,indent=2)
    handle.write('\n')
  return result_file


def run():
  '''
    Run a parameter sweep for the BSCQ SCD polar decoder.
  '''
  print(f'Running BSCQ SCD parameter sweep with {len(delta_values)} delta value(s) and {len(gamma_values)} gamma value(s)')
  results=[]
  experiment_index=0

  for delta in delta_values:
    for gamma in gamma_values:
      experiment_seed=seed+experiment_index
      result=run_single_experiment(
        delta,
        gamma,
        n,
        M,
        num_de_samples,
        experiment_seed,
        density_only,
      )
      results.append(result)
      print(summarize_result(result))
      experiment_index=experiment_index+1

  payload={
    'created_at': int(time.time()),
    'script': 'BSCQ_SCD_Parameter_Sweep.py',
    'description': 'BSCQ polar density-evolution and optional decoder Monte Carlo parameter sweep',
    'results': results,
  }
  output_path=write_results(payload,output_dir,output_file)
  print('Wrote sweep results to:', output_path)


if __name__ =="__main__":
  parser = ap.ArgumentParser('Run BSCQ SCD polar decoder parameter sweeps')
  parser.add_argument('--verbose', '-v', help='Display text output', action="store_true")
  parser.add_argument('-d', dest='delta', type=float, default=0.05, help='Delta')
  parser.add_argument('-g', dest='gamma', type=float, default=0.15, help='gamma')
  parser.add_argument('--delta-values', dest='delta_values', type=str, default=None, help='Comma-separated delta sweep values')
  parser.add_argument('--gamma-values', dest='gamma_values', type=str, default=None, help='Comma-separated gamma sweep values')
  parser.add_argument('-n', dest='n', type=int, default=3, help='polar stages')
  parser.add_argument('-M', dest='M', type=int, default=100, help='Number of Blocks')
  parser.add_argument('-nde', dest='num_de_samples', type=int, default=100, help='Number of DE samples')
  parser.add_argument('-s','--seed', dest='seed', type=int, default=None, help='Seed for RNG')
  parser.add_argument('--density-only', dest='density_only', help='Skip decoder Monte Carlo simulation', action="store_true")
  parser.add_argument('--output-dir', dest='output_dir', type=str, default='results', help='Directory for JSON output')
  parser.add_argument('--output-file', dest='output_file', type=str, default='bscq_scd_parameter_sweep.json', help='JSON output file name')

  args = parser.parse_args()

  if (args.seed is None):
    vars(args).update({"seed":int(time.time())%65536})

  delta_values=parse_float_list(args.delta_values) if args.delta_values else [args.delta]
  gamma_values=parse_float_list(args.gamma_values) if args.gamma_values else [args.gamma]
  vars(args).update({'delta_values':delta_values,'gamma_values':gamma_values})
  locals().update(vars(args))

  run()
