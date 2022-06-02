from synthesizer.hparams import hparams
from synthesizer.train import train,test
from utils.argutils import print_args
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str, help= \
        "Name for this model instance. If a model state from the same run ID was previously "
        "saved, the training will restart from there. Pass -f to overwrite saved states and "
        "restart from scratch.")
    parser.add_argument("syn_dir", type=str, default=argparse.SUPPRESS, help= \
        "Path to the synthesizer directory that contains the ground truth mel spectrograms, "
        "the wavs and the embeds.")
    parser.add_argument("-m", "--models_dir", type=str, default="synthesizer/saved_models/", help=\
        "Path to the output directory that will contain the saved model weights and the logs.")
    parser.add_argument("--hparams", default="",
                        help="Hyperparameter overrides as a comma-separated list of name=value "
							 "pairs")
    args = parser.parse_args()
    print_args(args, parser)

    args.hparams = hparams.parse(args.hparams)

    # Run the training
    test(**vars(args))