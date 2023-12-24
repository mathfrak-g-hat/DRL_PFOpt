# AJ Zerouali, 2023/04/11
'''
    This is the __init__ for the submodule:
    drl_pfopt.sb3_extras.sac.

This submodule of drl_pfopt contains modifications of the
    Stable Baselines 3 implementation of the SAC algorithm/policies. 
    For details, see:
    https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/sac
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/__init__.py
'''

# Default
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.sac.sac import SAC

# Extras
from drl_pfopt.sb3_extras.sac.sac_dropout_policy import DropOutMlpPolicy


__all__ = ["CnnPolicy", "MlpPolicy", "MultiInputPolicy", "DropOutMlpPolicy", "SAC"]