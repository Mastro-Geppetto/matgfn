#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import logging

# Path to zeo++-0.3/network
NETWORK=os.getenv('NETWORK', '')
if not NETWORK:
    print("Set path for zeo++-0.3/network")
    exit(1)

run_name = os.getenv("RUN_NAME", "")
if run_name == "":
    print("RUN_NAME environment variable is not set.")
    exit(1)

import sys
sys.path.append("./src")

from logging.handlers import RotatingFileHandler
# each run will have a unique log file with datetime stamp
time_now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")
log_file_name = run_name + '_agent_training.log'
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
    level=logging.INFO,#DEBUG,
    format=format,
    handlers=[debug_handler]  # Add the handler to basicConfig
)
logger = logging.getLogger(__name__)

import os
from matgfn.gflow.environments.sequence import SequenceEnvironment
from matgfn.gflow.agent import TrajectoryBalanceGFlowNet
from matgfn.gflow.flow_models.lstm import LSTM
from matgfn.reticular import PormakeStructureBuilder

import subprocess

import math
import torch

logger.info("starting main")

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

    mof=builder.make_pormake_mof(sequence)

    name_root=run_name+'temp'
    cif_name=name_root+'.cif'
    vol_name=name_root+'.vol'
    
    mof.write_cif(cif_name)
    logger.debug(f"wrote {cif_name}")

    if os.path.exists(cif_name)==False:
        return 0

    command=([NETWORK] + ['-vol'] + ['1.525'] + ['1.525'] + ['2000'] + [cif_name])
    subprocess.run(command,stdout=subprocess.DEVNULL)

    if os.path.exists(vol_name) == False:
        return 0

    lines=[]
    with open(vol_name) as result_file:
        for line in result_file:
            lines.append(line.rstrip())
    result_file.close()
    logger.debug(f"wrote {vol_name}")

    if len(lines)==0:
        return 0

    frags=lines[0].split()
    NAV=float(frags[17])
    AV=float(frags[11])

    command=(['rm'] + [vol_name])
    subprocess.run(command)

    area = AV + NAV
    
    return area

def surface_area_reward(sequence, builder):
    assert sequence[-1] == "[TER]"

    mof=builder.make_pormake_mof(sequence)

    name_root=run_name+'temp'
    cif_name=name_root+'.cif'
    sa_name=name_root+'.sa'
    
    mof.write_cif(cif_name)
    logger.debug(f"wrote {cif_name}")

    if os.path.exists(cif_name)==False:
        return 0

    command=([NETWORK] + ['-sa'] + ['1.525'] + ['1.525'] + ['2000'] + [cif_name])
    subprocess.run(command,stdout=subprocess.DEVNULL)

    if os.path.exists(sa_name) == False:
        return 0

    lines=[]
    with open(sa_name) as result_file:
        for line in result_file:
            lines.append(line.rstrip())
    result_file.close()
    logger.debug(f"wrote {sa_name}")

    if len(lines)==0:
        return 0

    frags=lines[0].split()
    NASA=float(frags[17])
    ASA=float(frags[11])

    command=(['rm'] + [sa_name])
    subprocess.run(command)

    area = ASA + NASA
    
    return area

def build_agent(builder,cutoff):

    token_vocabulary=builder.token_vocabulary
    n_slots=builder.n_slots
    mask=builder.mask

    env = SequenceEnvironment(
        token_vocabulary=token_vocabulary,
        termination_token="[TER]", 
        reward_function=lambda s: property_to_reward(
                                        surface_area_reward(s, builder) +volume_area_reward(s, builder)
                                        ,cutoff),
        mask=mask,
        render_function=None,
        max_sequence_length=n_slots, min_sequence_length=n_slots
        )
    
    flow_model =  LSTM(token_vocabulary=token_vocabulary, n_actions=env.action_space.n)
    agent = TrajectoryBalanceGFlowNet(env, flow_model)
    logger.info("agent created")
    return agent

def train_agent(builder, loss_threshold, run_name, cutoff):

    agent=build_agent(builder,cutoff)
    agent.train(True)

    current_loss=loss_threshold+99999

    all_observations=[]
    all_infos=[]
    all_rewards=[]
    all_losses=[]
    all_logZs=[]

    last_mean_loss=9999999

    continue_training=True

    logger.info("start training")
    while continue_training==True:
        logger.info("train..")

        observations, infos, rewards, losses, logZs = agent.fit(learning_rate=5e-3, num_episodes=5000, minibatch_size=5)

        all_observations+=observations
        all_infos+=infos
        all_rewards+=rewards
        all_losses+=losses
        all_logZs+=logZs

        test_losses=[]

        mean_loss_all_points=sum(losses)/len(losses)

        for loss in losses:
            if loss < (mean_loss_all_points*10):
                test_losses.append(loss)
                
        current_mean_loss=sum(test_losses)/len(test_losses)
        print('current loss =',current_mean_loss)
        logger.info(f'loss current {current_mean_loss} last {last_mean_loss} all loss count {len(all_losses)}')

        if current_mean_loss > last_mean_loss*0.95 and current_mean_loss < last_mean_loss:
            if current_mean_loss < 50:
                continue_training=False

        if current_mean_loss < loss_threshold:
            continue_training=False

        if len(all_losses) > 79999:
            continue_training=False

        last_mean_loss=current_mean_loss

    # extra 5000 episodes just to be sure of convergence
    observations, infos, rewards, losses, logZs = agent.fit(learning_rate=5e-3, num_episodes=5000, minibatch_size=5)

    all_observations+=observations
    all_infos+=infos
    all_rewards+=rewards
    all_losses+=losses
    all_logZs+=logZs

    agent_name=run_name+'_agent.pkl'
    os.makedirs('trained_agents', exist_ok=True)
    agent_path=os.path.join('trained_agents',agent_name)
    torch.save(agent.state_dict(), agent_path)
    logger.info(f'save agent at :{agent_path}')

    training_log_name=run_name+'_training_log.txt'
    os.makedirs('training_logs', exist_ok=True)
    training_log_path=os.path.join('training_logs',training_log_name)

    with open(training_log_path,'w') as f:
        
        for i in range(len(all_observations)):

            line='observation --- ' + str(all_observations[i]) + ' info --- ' + str(all_infos[i]) + ' reward --- ' + str(all_rewards[i]) + ' loss --- ' + str(all_losses[i]) + ' logZ --- ' + str(all_logZs[i])
            f.write(f"{line}\n")

    f.close()

    return 

topology_names=['tsg','cdl-e','cdz-e', 'eft', 'ffc', 'tff', 'asc', 'dmg', 'dnq', 'fso', 'urj']
cutoff=5000+500 # for both GSA & VSA
loss_cutoff=1.8

for name in topology_names:

    if run_name != name:
        continue

    logger.info(f'{name} no edges start')
    include_edges=False
    builder=PormakeStructureBuilder(topology_string=name,include_edges=include_edges)
    run_name=name+'_no_edges'
    train_agent(builder=builder,loss_threshold=loss_cutoff,run_name=run_name,cutoff=cutoff)
    logger.info(f'{name} no edges done')
        
    logger.info(f'{name} edges start')
    include_edges=True
    builder=PormakeStructureBuilder(topology_string=name,include_edges=include_edges)
    run_name=name+'_edges'
    train_agent(builder=builder,loss_threshold=loss_cutoff,run_name=run_name,cutoff=cutoff)
    logger.info(f'{name} edges done')
