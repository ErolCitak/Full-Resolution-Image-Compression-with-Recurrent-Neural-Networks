from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch
import os
from models import Encoder, Decoder, Binarizer
import numpy as np
from tensorboardX import SummaryWriter

def train(train_params, args, train_loader, val_loader):
    writer = SummaryWriter('log/{}'.format(args.model_name))
    encoder = Encoder().to(args.device)
    binarizer = Binarizer().to(args.device)
    decoder = Decoder().to(args.device)

    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr = train_params['lr'])
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr = train_params['lr'])
    binarizer_optimizer = torch.optim.Adam(binarizer.parameters(), lr = train_params['lr'])

    l1loss = torch.nn.L1Loss()

    best_loss = None

    for epoch in range(train_params['epochs']):
        train_loss = 0
        valid_loss = 0

        for batch_num, (sample_x, sample_y) in enumerate(train_loader):
            sample_x = sample_x.to(args.device)
            sample_y = sample_y.to(args.device)

            encoder_h1 = (torch.zeros(sample_x.size(0), 256, 64, 64).to(args.device), torch.zeros(sample_x.size(0), 256, 64, 64).to(args.device))
            encoder_h2 = (torch.zeros(sample_x.size(0), 512, 32, 32).to(args.device), torch.zeros(sample_x.size(0), 512, 32, 32).to(args.device))
            encoder_h3 = (torch.zeros(sample_x.size(0), 512, 16, 16).to(args.device), torch.zeros(sample_x.size(0), 512, 16, 16).to(args.device))

            decoder_h1 = (torch.zeros(sample_x.size(0), 512, 16, 16).to(args.device), torch.zeros(sample_x.size(0), 512, 16, 16).to(args.device))
            decoder_h2 = (torch.zeros(sample_x.size(0), 512, 32, 32).to(args.device), torch.zeros(sample_x.size(0), 512, 32, 32).to(args.device))
            decoder_h3 = (torch.zeros(sample_x.size(0), 256, 64, 64).to(args.device), torch.zeros(sample_x.size(0), 256, 64, 64).to(args.device))
            decoder_h4 = (torch.zeros(sample_x.size(0), 128, 128, 128).to(args.device), torch.zeros(sample_x.size(0), 128, 128, 128).to(args.device))

            losses = []
            enc_optimizer.zero_grad()
            binarizer_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            residual = sample_x
            for i in range(train_params['iterations']):
                print('==iteration:', i)
                # print('input:', residual.shape)
                x, encoder_h1, encoder_h2, encoder_h3 = encoder(residual, encoder_h1, encoder_h2, encoder_h3)
                x = binarizer(x)
                
                # print('\ncompressed:', x.shape, x.detach().numpy().astype(np.bool).nbytes)
                # print()
                output, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = decoder(x, decoder_h1, decoder_h2, decoder_h3, decoder_h4)
                # print('output:', output.shape)

                residual = residual - output
                losses.append(residual.abs().mean())


            loss = np.mean(losses)
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            binarizer_optimizer.step()

            train_loss += loss.item()

        for batch_num, (sample_x, sample_y) in enumerate(val_loader):
            sample_x = sample_x.to(args.device)
            sample_y = sample_y.to(args.device)

            encoder_h1 = (torch.zeros(sample_x.size(0), 256, 64, 64).to(args.device), torch.zeros(sample_x.size(0), 256, 64, 64).to(args.device))
            encoder_h2 = (torch.zeros(sample_x.size(0), 512, 32, 32).to(args.device), torch.zeros(sample_x.size(0), 512, 32, 32).to(args.device))
            encoder_h3 = (torch.zeros(sample_x.size(0), 512, 16, 16).to(args.device), torch.zeros(sample_x.size(0), 512, 16, 16).to(args.device))

            decoder_h1 = (torch.zeros(sample_x.size(0), 512, 16, 16).to(args.device), torch.zeros(sample_x.size(0), 512, 16, 16).to(args.device))
            decoder_h2 = (torch.zeros(sample_x.size(0), 512, 32, 32).to(args.device), torch.zeros(sample_x.size(0), 512, 32, 32).to(args.device))
            decoder_h3 = (torch.zeros(sample_x.size(0), 256, 64, 64).to(args.device), torch.zeros(sample_x.size(0), 256, 64, 64).to(args.device))
            decoder_h4 = (torch.zeros(sample_x.size(0), 128, 128, 128).to(args.device), torch.zeros(sample_x.size(0), 128, 128, 128).to(args.device))

            x, encoder_h1, encoder_h2, encoder_h3 = encoder(sample_x, encoder_h1, encoder_h2, encoder_h3)
            x = binarizer(x)
            output, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = decoder(x, decoder_h1, decoder_h2, decoder_h3, decoder_h4)

            valid_loss += l1loss(output, sample_x)


        print("Train Loss for epoch {}: {}".format(epoch, (train_loss / len(train_loader))))
        print("Validation Loss for epoch {}: {}".format(epoch, (valid_loss / len(val_loader))))

        if best_loss is None or valid_loss > best_loss:
            save_models(encoder, binarizer, decoder)
            best_loss = valid_loss

    writer.close()

    return enc, binarizer, decoder

def save_models(encoder, binarizer, decoder):
    torch.save(encoder.state_dict, 'save/{model_name}_e'.format(model_name=args.model_name))
    torch.save(binarizer.state_dict, 'save/{model_name}_b'.format(model_name=args.model_name))
    torch.save(decoder.state_dict, 'save/{model_name}_d'.format(model_name=args.model_name))
