import torch 

def load(model_path):
    """
    Load a model from a file.

    # Arguments
    model_path: str. The path to the file containing the model.

    # Returns
    model: Model. The loaded model.
    """
    model = torch.load(model_path)
    return model