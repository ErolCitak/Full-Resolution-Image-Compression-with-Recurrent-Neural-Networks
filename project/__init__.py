import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms import Resize, ToTensor, ToPILImage
from PIL import Image


# Binary output channel dimension
b_outchannel_dim = 125
# Number of iterations for encoding
n_iterations = 16

model_name = 'crnn_sig_lr001_125'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"device: {device}")

def get_hidden_layers(sample_x):
    encoder_h1 = (
        torch.zeros(sample_x.size(0), 256, 64, 64).to(device),
        torch.zeros(sample_x.size(0), 256, 64, 64).to(device)
    )
    encoder_h2 = (
        torch.zeros(sample_x.size(0), 512, 32, 32).to(device),
        torch.zeros(sample_x.size(0), 512, 32, 32).to(device))
    encoder_h3 = (
        torch.zeros(sample_x.size(0), 512, 16, 16).to(device),
        torch.zeros(sample_x.size(0), 512, 16, 16).to(device)
    )

    decoder_h1 = (
        torch.zeros(sample_x.size(0), 512, 16, 16).to(device),
        torch.zeros(sample_x.size(0), 512, 16, 16).to(device)
    )
    decoder_h2 = (
        torch.zeros(sample_x.size(0), 512, 32, 32).to(device),
        torch.zeros(sample_x.size(0), 512, 32, 32).to(device)
    )
    decoder_h3 = (
        torch.zeros(sample_x.size(0), 256, 64, 64).to(device),
        torch.zeros(sample_x.size(0), 256, 64, 64).to(device)
    )
    decoder_h4 = (
        torch.zeros(sample_x.size(0), 128, 128, 128).to(device),
        torch.zeros(sample_x.size(0), 128, 128, 128).to(device)
    )

    return encoder_h1, encoder_h2, encoder_h3, decoder_h1, decoder_h2, decoder_h3, decoder_h4


def encode(img, bottleneck):
    """
    Your code here
    img: a 256x256 PIL Image
    bottleneck: an integer from {4096,16384,65536}
    return: a numpy array less <= bottleneck bytes
    """
    input_transform = transforms.Compose([Resize((256, 256)), ToTensor()])
    sample_x = input_transform(img).unsqueeze(0).to(device)

    encoder_h1, encoder_h2, encoder_h3, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = \
        get_hidden_layers(sample_x)

    residual = sample_x
    for i in range(n_iterations):
        x, encoder_h1, encoder_h2, encoder_h3 = encoder(
            residual, encoder_h1, encoder_h2, encoder_h3)
        x = binarizer(x)
        if i != n_iterations - 1:
            output, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = decoder(
                x, decoder_h1, decoder_h2, decoder_h3, decoder_h4)
            residual = sample_x - output

    x = x.to(torch.device('cpu'))
    codes = x.detach().numpy().astype(np.int8)
    codes = (codes + 1) // 2
    codes = np.packbits(codes, axis=1)
    print(f"nbytes: {codes.nbytes}")
    return codes
    
def decode(x, bottleneck):
    """
    Your code here
    x: a numpy array
    bottleneck: an integer from {4096,16384,65536}
    return a 256x256 PIL Image
    """
    x = np.unpackbits(x, axis=1, count=-(8 - b_outchannel_dim % 8) % 8).astype(np.float32)
    x = x * 2 - 1
    x = torch.Tensor(x).to(device)

    output = decoder.forward(x, decoder_h1, decoder_h2, decoder_h3, decoder_h4)[0]
    output = output.squeeze(0)
    output = output.detach().cpu().numpy().transpose((1,2,0))
    output = Image.fromarray(np.uint8(output * 255))
    output.save('reconst.jpg')
    return output


"""
Loading in Model
"""
from .models import Encoder, Binarizer, Decoder

encoder = Encoder().to(device)
binarizer = Binarizer().to(device)
decoder = Decoder().to(device)

# Load model weights here
if torch.cuda.is_available():
    encoder.load_state_dict(
        torch.load('project/save/{}_e.pth'.format(model_name))['model_state_dict'])
    binarizer.load_state_dict(
        torch.load('project/save/{}_b.pth'.format(model_name))['model_state_dict'])
    decoder.load_state_dict(
        torch.load('project/save/{}_d.pth'.format(model_name))['model_state_dict'])
else:
    encoder.load_state_dict(
        torch.load('project/save/{}_e.pth'.format(model_name), map_location='cpu')['model_state_dict'])
    binarizer.load_state_dict(
        torch.load('project/save/{}_b.pth'.format(model_name), map_location='cpu')['model_state_dict'])
    decoder.load_state_dict(
        torch.load('project/save/{}_d.pth'.format(model_name), map_location='cpu')['model_state_dict'])

encoder.eval()
binarizer.eval()
decoder.eval()
