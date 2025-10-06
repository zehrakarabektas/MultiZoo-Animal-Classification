import torch
import timm

def load_model(path):
    model = timm.create_model("beit_base_patch16_224", pretrained=False, num_classes=90)  
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    return model
