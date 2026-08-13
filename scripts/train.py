#!/usr/bin/env python
import argparse
from graphtraffic.utils.io import load_yaml
from graphtraffic.experiments.train import run_training
p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/smoke.yaml"); a=p.parse_args(); path=run_training(load_yaml(a.config)); print(path)
