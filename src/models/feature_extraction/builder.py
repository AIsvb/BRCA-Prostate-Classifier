import src.models.feature_extraction.modelzoo as zoo
from src.utils.constants import MODEL2CONSTANTS

def build_fe_from_name(name, train=False, checkpoint=True):

    # -- check if model exists in the model zoo
    assert(name in zoo.__dict__)

    # -- construct model
    constants = MODEL2CONSTANTS[name]
    model = zoo.__dict__[name](checkpoint=constants['chkpt_dir'] if checkpoint else None)

    # -- specify whether gradient computation is required
    for param in model.parameters():
        param.requires_grad = train
    
    return model, constants
