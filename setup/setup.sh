# AJ Zerouali, 23/09/13
# NOTE: Ensure this script has execution permission (chmod).
#       Pip does not resolve conflicts if packages installed 
#       from requirements file. Script also patches PyFolio.

echo "########################################################"
echo "########### Setting-up DRL_PFOpt Environment ###########"
echo "########################################################"
echo ""

echo "###########################"
echo "##### Updating PIP... #####"
echo "###########################"
pip install --upgrade pip
echo ""

echo "###############################"
echo "##### Updating PyTorch... #####"
echo "###############################"
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
echo ""

# Protobuf (avoiding package conflicts)
pip install protobuf==3.19.2

# Linux
yes | sudo apt update
echo "-> Installing python-opengl, ffmpeg, xvfb, swig..."
yes | sudo apt install python-opengl ffmpeg xvfb swig tree

echo "################################################"
echo "##### Installing DRL_PFOpt dependencies... #####"
echo "################################################"

# Gym dependencies
pip install swig cmake pyopengl pygame mujoco box2d-py moviepy pyvirtualdisplay

# Gym
pip install gym

# Stable baselines - Carlos Luis fork (v2.0.0a0)
pip install -U git+https://github.com/carlosluis/stable-baselines3@fix_tests

# Data APIs
## Note (23/08/21):  Issue with pandas-market-calendars v4.2.0, downrgade to 4.1.4 temporarily.
## Note (23/08/31):  1) Pandas 2.1.0 breaks drl_pfopt. Keep 1.5.3 because 2.0.0 incompatible with pytorch_forecasting.
##                   2) Downgrade alpaca-py to 0.8.2
pip install pandas==1.5.3 alpaca-py==0.8.2 yahoofinancials yfinance pandas-market-calendars==4.1.4
pip install exchange_calendars==3.6.3 # Because of a bug encountered previously

# Portfolio analysis
pip install pyfolio PyPortfolioOpt stockstats 
#pip install tensorboard==2.13

echo "###############################"
echo "##### Patching pyfolio... #####"
echo "###############################"
#cd 
cp /notebooks/setup/Patches/pyfolio/timeseries.py /usr/local/lib/python3.9/dist-packages/pyfolio/timeseries.py
echo "pyfolio/timeseries.py has been updated..."
cp /notebooks/setup/Patches/pyfolio/plotting.py /usr/local/lib/python3.9/dist-packages/pyfolio/plotting.py
echo "pyfolio/plotting.py has been updated..."
echo ""


echo "#############################"
echo "##### Install complete. #####"
echo "#############################"
