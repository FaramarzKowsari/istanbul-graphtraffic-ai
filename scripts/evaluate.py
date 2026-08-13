#!/usr/bin/env python
import argparse, json
from graphtraffic.utils.io import load_yaml
from graphtraffic.experiments.evaluate import run_evaluation
p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/smoke.yaml"); a=p.parse_args(); print(json.dumps(run_evaluation(load_yaml(a.config),),indent=2))
