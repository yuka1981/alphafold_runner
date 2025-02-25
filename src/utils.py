
import os

def setup_environment(args):
    """
    Set up environment variables based on the system configuration
    """
    # TODO: add the following environment variables flexibly
    # os.environ["TF_DETERMINISTIC_OPS"] = 0
    # os.environ["TF_FORCE_UNIFIED_MEMORY"] = 0
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = 0.9
    
    match args.platform:
        case "amd":
            os.environ["JAX_PLATFORMS"] = "rocm"
            os.environ["ROCM_PATH"] = "/opt/rocm"
            match args.mode:
                case "nogpu":
                    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
                case "1gpu0":
                    os.environ["HIP_VISIBLE_DEVICES"] = "0"
                case "1gpu1":
                    os.environ["HIP_VISIBLE_DEVICES"] = "1"
                case "2gpus":
                    os.environ["HIP_VISIBLE_DEVICES"] = "0,1"
                case "all_gpus":
                    os.environ["HIP_VISIBLE_DEVICES"] = "0,1"
                case _:
                    raise ValueError(f"Invalid mode {args.mode}")
        
        case "nvidia":
            os.environ["JAX_PLATFORMS"] = "cuda"
            match args.mode:
                case "nogpu":
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                case "1gpu0":
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                case "1gpu1":
                    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
                case "2gpus":
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
                case "all_gpus":
                    # TODO: better to check number of GPUs
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
                case _:
                    raise ValueError(f"Invalid mode {args.mode}")
        
        case _:
            raise ValueError(f"Invalid platform {args.platform}")



# check module command
# module purge
# check system modules
# module load
    # match case via platform

# setup tf environment

# log message

## log message related ##
# print results

## msa realted ##
# copy msa files to output directory
# copy msa files automatically when nomsa is setup


