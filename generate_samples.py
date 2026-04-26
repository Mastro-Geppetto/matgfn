#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Generate observations from trained models
'''

cutoff=5000+500 # for both GSA & VSA
max_observations=160265
max_zero_reward_percentage=0.05

from datetime import datetime
import os
import logging
import csv

import optparse

# create option parser to allow setting run_name, max_observations and with_edges from command line
parser = optparse.OptionParser()
parser.add_option("--run_name", dest="run_name", help="Name of the run")
parser.add_option("--max_observations", dest="max_observations", help="Maximum number of observations to generate")
parser.add_option("--use_with_edges_model", action="store_true", dest="with_edges", help="Whether to use with edges model in the generated observations")
parser.add_option("--use_without_edges_model", action="store_true", dest="without_edges", help="Whether to use without edges model in the generated observations")

options, args = parser.parse_args()

run_name = options.run_name if options.run_name else os.getenv("RUN_NAME", "")
if run_name == "":
    print("RUN_NAME environment variable is not set.")
    exit(1)

max_observations = int(options.max_observations) if options.max_observations else int(os.getenv("MAX_OBSERVATIONS", max_observations))

if not options.with_edges and not options.without_edges:
    print("Please specify at least one model to use for generating observations: --use_with_edges_model and/or --use_without_edges_model")
    exit(1)

import sys
sys.path.append("./src")

from logging.handlers import RotatingFileHandler
# each run will have a unique log file with datetime stamp
time_now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")
log_file_name = '_gen_' + time_now + '.log'
if options.with_edges:
    log_file_name = run_name + '_with_edges' + log_file_name
elif options.without_edges:
    log_file_name = run_name + '_withOUT_edges' + log_file_name
# Create a rotating file handler
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 100    # Keep 100 backup files
format = '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
debug_handler = RotatingFileHandler(
    filename=log_file_name,
    maxBytes=MAX_BYTES,
    backupCount=BACKUP_COUNT
)
logging.basicConfig(
    level=logging.INFO,  # DEBUG,
    format=format,
    handlers=[debug_handler]  # Add the handler to basicConfig
)
logger = logging.getLogger(__name__)

# Path to zeo++-0.3/network
NETWORK = os.getenv('NETWORK', '')
if not NETWORK:
    print("Set path for zeo++-0.3/network")
    logger.error("Set path for zeo++-0.3/network")
    exit(1)


import os
from matgfn.gflow.environments.sequence import SequenceEnvironment
from matgfn.gflow.agent import TrajectoryBalanceGFlowNet
from matgfn.gflow.flow_models.lstm import LSTM
from matgfn.reticular import PormakeStructureBuilder

import subprocess

import math
import torch
from tqdm import tqdm


def property_to_reward(prop, cutoff):
    if prop < cutoff:
        return 0
    else:
        return math.exp((prop - cutoff) / cutoff)


def reward_to_property(reward, cutoff):
    if reward == 0:
        return 0
    else: 
        return math.log(reward) * cutoff + cutoff


def volume_area_reward(sequence, builder):
    assert sequence[-1] == "[TER]"
    #
    mof = builder.make_pormake_mof(sequence)
    #
    name_root = run_name+'temp'
    cif_name = name_root+'.cif'
    vol_name = name_root+'.vol'
    # write cif file
    mof.write_cif(cif_name)
    logger.debug(f"wrote {cif_name}")
    if os.path.exists(cif_name) is False:
        return 0
    command = ([NETWORK] + ['-vol'] + ['1.525'] + ['1.525'] + ['2000'] + [cif_name])
    subprocess.run(command, stdout=subprocess.DEVNULL)
    #
    if os.path.exists(vol_name) == False:
        return 0
    #
    lines = []
    with open(vol_name) as result_file:
        for line in result_file:
            lines.append(line.rstrip())
    result_file.close()
    logger.debug(f"wrote {vol_name}")
    #
    if len(lines) == 0:
        return 0
    #
    frags = lines[0].split()
    NAV = float(frags[17])
    AV = float(frags[11])
    #
    command = (['rm'] + [vol_name])
    subprocess.run(command)
    #
    area = AV + NAV
    return area


def surface_area_reward(sequence, builder):
    assert sequence[-1] == "[TER]"
    #
    mof = builder.make_pormake_mof(sequence)
    #
    name_root = run_name+'temp'
    cif_name = name_root+'.cif'
    sa_name = name_root+'.sa'
    # write cif file
    mof.write_cif(cif_name)
    logger.debug(f"wrote {cif_name}")
    #
    if os.path.exists(cif_name) is False:
        return 0
    #
    command = ([NETWORK] + ['-sa'] + ['1.525'] + ['1.525'] + ['2000'] + [cif_name])
    subprocess.run(command, stdout=subprocess.DEVNULL)
    #
    if os.path.exists(sa_name) == False:
        return 0
    #
    lines = []
    with open(sa_name) as result_file:
        for line in result_file:
            lines.append(line.rstrip())
    result_file.close()
    logger.debug(f"wrote {sa_name}")
    #
    if len(lines)==0:
        return 0
    #
    frags = lines[0].split()
    NASA = float(frags[17])
    ASA = float(frags[11])
    #
    command = (['rm'] + [sa_name])
    subprocess.run(command)
    #
    area = ASA + NASA
    return area


def build_agent(builder, cutoff):
    token_vocabulary = builder.token_vocabulary
    n_slots = builder.n_slots
    mask = builder.mask
    #
    env = SequenceEnvironment(
        token_vocabulary=token_vocabulary,
        termination_token="[TER]",
        reward_function=lambda s: property_to_reward(
                                        surface_area_reward(
                                            s, builder) + volume_area_reward(
                                                s, builder), cutoff),
        mask=mask,
        render_function=None,
        max_sequence_length=n_slots, min_sequence_length=n_slots
        )
    #
    flow_model = LSTM(
                    token_vocabulary=token_vocabulary,
                    n_actions=env.action_space.n)
    agent = TrajectoryBalanceGFlowNet(env, flow_model)
    logger.info("agent created")
    return agent


def write_observations(save_name, observations):
    os.makedirs('trained_observations', exist_ok=True)
    file_name = os.path.join(
        'trained_observations', save_name+'_observations.csv')
    with open(file_name, "a", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(["sequence", "score"])  # optional header
        for seq, score in observations:
            writer.writerow([" ".join(seq), score])  # sequence as one cell


def generate_observations(save_name, agent, max_observations):
    trained_observations = []
    for i in tqdm(range(0, max_observations)):
        obs, info, reward, _ = agent.sample()
        sequence = agent.env._sequence[:-1]
        trained_observations.append([sequence, 0 if reward == 0 else math.log(reward) * cutoff + cutoff])
        if len(trained_observations) >= 1000:
            write_observations(
                save_name,
                trained_observations)
            trained_observations = []
    # write remaining observations if any
    write_observations(save_name, trained_observations)

logger.info("starting main")

if options.with_edges:
    logger.info(f'starting to generate observations for {run_name}_with_edges_observations')
    builder_with_edges = PormakeStructureBuilder(
                            topology_string=run_name,
                            include_edges=True)
    agent_with_edges = build_agent(builder_with_edges, cutoff)
    agent_state_with_edges = torch.load(
                                os.path.join(
                                    'trained_agents',
                                    run_name+'_edges_agent.pkl'))
    # 1. load model
    agent_with_edges.load_state_dict(agent_state_with_edges)
    # 2. eval mode
    agent_with_edges.eval()
    # 3. generate observations
    generate_observations(
        run_name+'_with_edges_observations',
        agent_with_edges,
        max_observations)
    logger.info(f'finished generating observations for {run_name}_with_edges_observations')

if options.without_edges:
    logger.info(f'starting to generate observations for {run_name}_without_edges_observations')
    builder_without_edges = PormakeStructureBuilder(
                            topology_string=run_name,
                            include_edges=False)
    agent_without_edges = build_agent(builder_without_edges, cutoff)
    agent_state_without_edges = torch.load(
                                    os.path.join(
                                        'trained_agents',
                                        run_name+'_no_edges_agent.pkl'))
    # 1. load model
    agent_without_edges.load_state_dict(agent_state_without_edges)
    # 2. eval mode
    agent_without_edges.eval()
    # 3. generate observations
    generate_observations(
        run_name+'_without_edges_observations',
        agent_without_edges,
        max_observations)
    logger.info(f'finished generating observations for {run_name}_without_edges_observations')
