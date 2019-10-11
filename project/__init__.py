import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms import Resize, ToTensor, ToPILImage
from PIL import Image


# Number of iterations for encoding
n_iterations = 16
size = 256
batch_size = 1

model_names = ['crnn_128_stoch',
               'clstm_sigmoid_stochastic_scheduler_512_1k',
               'clstm_sigmoid_stochastic_scheduler_512_1k']
out_channels_b = [128, 512, 512]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"device: {device}")


def get_models(bottleneck):
    if bottleneck == 4096:
        encoder = encoders[0]
        binarizer = binarizers[0]
        decoder = decoders[0]
    elif bottleneck == 16384:
        encoder = encoders[1]
        binarizer = binarizers[1]
        decoder = decoders[1]
    else:
        encoder = encoders[2]
        binarizer = binarizers[2]
        decoder = decoders[2]
    return encoder, binarizer, decoder

def encode(img, bottleneck):
    """
    Your code here
    img: a 256x256 PIL Image
    bottleneck: an integer from {4096,16384,65536}
    return: a numpy array less <= bottleneck bytes
    """
    encoder, binarizer, decoder = get_models(bottleneck)
    input_transform = transforms.Compose([Resize((256, 256)), ToTensor()])
    sample_x = input_transform(img).unsqueeze(0).to(device)

    encoder.init_hidden(device)
    decoder.init_hidden(device)

    residual = sample_x
    for i in range(n_iterations):
        x = encoder(residual)
        x = binarizer(x)
        if i != n_iterations - 1:
            output = decoder(x)
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
    encoder, binarizer, decoder = get_models(bottleneck)
    x = np.unpackbits(x, axis=1).astype(np.float32)
    x = x * 2 - 1
    x = torch.Tensor(x).to(device)

    output = decoder(x)
    output = output.squeeze(0)
    output = output.detach().cpu().numpy().transpose((1,2,0))
    output = Image.fromarray(np.uint8(output.clip(0, 1) * 255))
    output.save('reconst.jpg')
    return output


"""
Loading in Model
"""
from .models2 import Encoder, Binarizer, Decoder

encoders = []
binarizers = []
decoders = []

# Load model weights here
for i, model_name in enumerate(model_names):
    encoders.append(Encoder(size, batch_size).to(device))
    binarizers.append(Binarizer(out_channels_b[i]).to(device))
    decoders.append(Decoder(size, batch_size, out_channels_b[i]).to(device))

    if torch.cuda.is_available():
        encoders[i].load_state_dict(
            torch.load('project/save/{}_e.pth'.format(model_name))['model_state_dict'])
        binarizers[i].load_state_dict(
            torch.load('project/save/{}_b.pth'.format(model_name))['model_state_dict'])
        decoders[i].load_state_dict(
            torch.load('project/save/{}_d.pth'.format(model_name))['model_state_dict'])
    else:
        encoders[i].load_state_dict(
            torch.load('project/save/{}_e.pth'.format(model_name), map_location='cpu')['model_state_dict'])
        binarizers[i].load_state_dict(
            torch.load('project/save/{}_b.pth'.format(model_name), map_location='cpu')['model_state_dict'])
        decoders[i].load_state_dict(
            torch.load('project/save/{}_d.pth'.format(model_name), map_location='cpu')['model_state_dict'])

    encoders[i].eval()
    binarizers[i].eval()
    decoders[i].eval()
