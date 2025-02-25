import os
# import sys
# import argparse

SUPPORTED_MODES = {"nogpu", "1gpu0", "1gpu1", "2gpus", "all_gpus"}
SUPPORTED_PRESETS = {"monomer", "multimer"}
SUPPORTED_PLATFORMS = {"cpu", "nvidia", "amd"}
MSA_OPTIONS = {"msa", "nomsa"}

def validate_arguments(args):
    """
    Validate the arguments passed to the program
    """
    if not os.path.isfile(args.input_fasta):
        raise FileNotFoundError(f"Input fasta file {args.input_fasta} does not exist")
    if args.mode not in SUPPORTED_MODES:
        raise ValueError(f"Invalid mode {args.mode}, must be one of {SUPPORTED_MODES}")
    if args.preset not in SUPPORTED_PRESETS:
        raise ValueError(f"Invalid preset {args.preset}, must be one of {SUPPORTED_PRESETS}")
    if args.platform not in SUPPORTED_PLATFORMS:
        # sys.exit(f"Invalid platform {args.platform}")
        raise ValueError(f"Invalid platform {args.platform}, must be one of {SUPPORTED_PLATFORMS}")
    return True
