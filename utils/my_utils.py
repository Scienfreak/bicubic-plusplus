import torch
import torch.nn as nn


# Define a function to save the model weights to a text file
def save_weights_to_txt(model, txt_file):
    with open(txt_file, 'w') as f:
        for name, param in model.named_parameters():
            if 'weight' in name:
                f.write(f'Layer: {name}\n')
                f.write(str(param.data.cpu().numpy().astype('float16')))
                f.write('\n\n')