import os
import argparse
import time
import mlflow


def evalfunc(a, b):
    return a**2 + b**2


def main(in1, in2):
    mlflow.set_experiment("Demo Experiment")
    mlflow.start_run()
    mlflow.log_param('param 1', in1)
    mlflow.log_param('param 2', in1)

    metric = evalfunc(in1, in2)

    mlflow.log_metric('metric', metric)

    #os.mkdir('info')
    with open('info/example_artifact.txt', 'w') as file:
        file.write(f"Artifact Created at Time: {time.asctime()}")
    mlflow.log_artifact("../week-06/info")


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--param1", "-p1", type=int, default=5)
    args.add_argument("--param2", "-p2", type=int, default=5)
    parsed_args = args.parse_args()

    main(parsed_args.param1, parsed_args.param2)