# Official Repository for "Cognitive Graph-based Human–Machine Teaming for Distributed Situational Awareness in Multi-Agent Autonomous Driving"


We model cooperative multi-agent driving with a human supervisor using CTDE, where each autonomous vehicle maintains an individual mental model with belief, evidential intent, and expectations over peers.
A fully directed trust graph combines performance, agreement, and uncertainty to adapt trust over time, while discrete human feedback (confirm/deny/monitor etc.) nudges trust and modulates learning and coordination.
Implemented using the macad-gym library.

[![PyPI version fury.io](https://badge.fury.io/py/macad-gym.svg)](https://pypi.python.org/pypi/macad-gym/)
[![PyPI format](https://img.shields.io/pypi/pyversions/macad-gym.svg)](https://pypi.python.org/pypi/macad-gym/)
[![Downloads](https://pepy.tech/badge/macad-gym)](https://pepy.tech/project/macad-gym)
### Quick Start

Install MACAD-Gym using `pip install macad-gym`.
 If you have `CARLA_SERVER` setup, you can get going using the following 3 lines of code. If not, follow the
[Getting started steps](#getting-started).

#### Training RL Agents

```python
import gym
import macad_gym
env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")

# Your agent code here
```

 Any RL library that supports the OpenAI-Gym API can be used to train agents in MACAD-Gym. The [MACAD-Agents](https://github.com/praveen-palanisamy/macad-agents) repository provides sample agents as a starter.

#### Visualizing the Environment

To test-drive the environments, you can run the environment script directly. For example, to test-drive the `HomoNcomIndePOIntrxMASS3CTWN3-v0` environment, run:

```bash
python -m macad_gym.envs.homo.ncom.inde.po.intrx.ma.stop_sign_3c_town03
```


### Supported Platforms

MACAD-Gym and CG-HMT have been tested on Windows, macOS, and Linux. The Python installation and training commands are the same across platforms, but the CARLA server path differs:

- **Windows:** Set the CARLA_SERVER environment variable as already shown.
- **macOS/Linux:** Point `CARLA_SERVER` to the shell script used to launch CARLA, for example:
  ```bash
  export CARLA_SERVER="/path/to/CARLA/CarlaUE4.sh"
  ```
  Ensure the shell has execute permission (`chmod +x`) and include the folder in `PATH` if needed.

You can run the `python -m macad_gym.envs...` command shown above on all supported systems once the CARLA server is running locally.

## Setup

Create and activate a new conda environment:

```bash
conda create --name cg-hmt python=3.7.16
conda activate cg-hmt
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and set the CARLA server path (for Widnows):

```powershell
$env:CARLA_SERVER="G:\CARLA\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe"
```

For macOS/Linux, use the corresponding server launch script and export the path before training:
```bash
export CARLA_SERVER="/path/to/CARLA/CarlaUE4.sh"
```

Run our proposed CG-HMT approach:

```bash
python CG-HMT-PPO.py --num-iters 300 --train-batch-size 12000 --rollout-fragment-length 200
```
