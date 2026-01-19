#- Command Line Interface.
import argparse
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train.train_model import training_pipeline
from src.routes import router
from src.data_pipeline.data_preparation import data_pipeline

"""print("train config: ",config_utils.train_config())
print("train config: ",config_utils.data_config())
print("train config: ",config_utils.paths_config())
print("train config: ",config_utils.inference_config())"""

def main():
    parser = argparse.ArgumentParser(description="House Price Prediction CLI")
    parser.add_argument("command", choices=["preprocess", "train", "route"],
                        help="Choose an action: preprocess, train, or route")

    args = parser.parse_args()

    if args.command == "preprocess":
        data_pipeline()
    elif args.command == "train":
        training_pipeline()
    elif args.command == "route":
        router.run_route()

if __name__ == "__main__":
    main()
