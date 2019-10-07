import numpy as np
import PIL


def encode(img, bottleneck):
    """
    Your code here
    img: a 256x256 PIL Image
    bottleneck: an integer from {4096,16384,65536}
    return: a numpy array less <= bottleneck bytes
    """
    input_transform = transforms.Compose([Resize((256, 256)), ToTensor()])
    img_tensor = input_transform(img)

    return binarizer.forward(encoder(img_tensor)).detach().numpy()
    
def decode(x, bottleneck):
    """
    Your code here
    x: a numpy array
    bottleneck: an integer from {4096,16384,65536}
    return a 256x256 PIL Image
    """
    output = decoder.forward(torch.Tensor(x))
    output_transform = transforms.Compose([Resize((256, 256)), ToTensor()])

    return output_transform(output)



"""
Loading in Model
"""
from .models import Encoder, Binarizer, Decoder

encoder = Encoder()
binarizer = Binarizer()
decoder = Decoder()

encoder.eval()
binarizer.eval()
decoder.eval()

# Load model weights here
encoder.load_state_dict(torch.load('weights/encoder.pth', map_location='cpu'))
binarizer.load_state_dict(torch.load('weights/binarizer.pth', map_location='cpu'))
decoder.load_state_dict(torch.load('weights/decoder.pth', map_location='cpu'))

