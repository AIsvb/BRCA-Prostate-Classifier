from dotenv import load_dotenv
import os

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]
ZERO_MEAN = [0., 0., 0.]
UNIT_STD = [1., 1., 1.]

# Load .env file from the repository root
load_dotenv()  # Automatically looks for .env in the current or parent directories
ROOT = os.environ.get('CHECKPOINT_DIR')

MODEL2CONSTANTS = {
    "vith14UNI2": {
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD, 
        "chkpt_dir": ROOT + '/uni2_checkpoint.pth.tar',
        "embed_dim": 1536
    },
    "optimus": {
        "mean": [0.707223, 0.578729, 0.703617],
        "std": [0.211883, 0.230117, 0.177517],
        "chkpt_dir": "",
        "embed_dim": 1536
    },
    "virchow2": {
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "chkpt_dir": "",
        "embed_dim": 2560
    },
    "vitGigaPath": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD,
        "chkpt_dir": ROOT + '/provgigapath_checkpoint.pth.tar',
        "embed_dim": 1536
    },
    "longnetGigaPath": {
        "chkpt_dir": ROOT + '/provgigapath_slide_encoder.pth.tar'
    }
}