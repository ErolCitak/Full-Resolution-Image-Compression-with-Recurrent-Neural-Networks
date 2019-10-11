import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as LS
from PIL import Image
from torchvision import transforms
from tensorboardX import SummaryWriter

from models2 import Encoder, Decoder, Binarizer


def img_normalize(imgs):
    return (imgs+1.0)/2

def save_models(args, encoder, binarizer, decoder, epoch, enc_optimizer, dec_optimizer, binarizer_optimizer, loss):
    path = "save/"
    torch.save({
            'epoch': epoch,
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': enc_optimizer.state_dict(),
            'loss': loss
            }, path + args.model_name+f"_e.pth")

    torch.save({
            'epoch': epoch,
            'model_state_dict': binarizer.state_dict(),
            'optimizer_state_dict': binarizer_optimizer.state_dict(),
            'loss': loss
            }, path + args.model_name+f"_b.pth")
    torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': dec_optimizer.state_dict(),
            'loss': loss
            }, path + args.model_name+f"_d.pth")


def train(train_params, args, train_loader, val_loader):
    encoder = Encoder(args.size, train_params['batch_size']).to(args.device)
    binarizer = Binarizer(args.out_channels_b, args.stochastic).to(args.device)
    decoder = Decoder(args.size, train_params['batch_size'], args.out_channels_b).to(args.device)

    enc_optimizer = torch.optim.Adam(
        encoder.parameters(), lr=train_params['lr'])
    dec_optimizer = torch.optim.Adam(
        decoder.parameters(), lr=train_params['lr'])
    binarizer_optimizer = torch.optim.Adam(
        binarizer.parameters(), lr=train_params['lr'])
    enc_scheduler = LS.MultiStepLR(
        enc_optimizer, milestones=[3, 10, 20, 50, 100], gamma=0.5)
    dec_scheduler = LS.MultiStepLR(
        dec_optimizer, milestones=[3, 10, 20, 50, 100], gamma=0.5)
    binarizer_scheduler = LS.MultiStepLR(
        binarizer_optimizer, milestones=[3, 10, 20, 50, 100], gamma=0.5)

    l1loss = torch.nn.L1Loss()

    best_loss = float('inf')
    best_encoder, best_binarizer, best_decoder = None, None, None
    full_patience = 10
    patience = full_patience
    batch_size = train_params['batch_size']
    writer = SummaryWriter('log/{}'.format(args.model_name))
    log_interval = int(len(train_loader) / batch_size * 0.05)
    val_interval = int(len(train_loader) / batch_size)
    print('log_interval:', log_interval, 'val_interval:', val_interval)

    for epoch in range(train_params['epochs']):
        print('== Epoch:', epoch)
        epoch_loss = 0
        curr_loss = 0
        for batch_idx, (sample_x, sample_y) in enumerate(train_loader):
            print(f"batch_idx: {batch_idx}")
            sample_x = sample_x.to(args.device)
            sample_y = sample_y.to(args.device)

            encoder.init_hidden(args.device)
            decoder.init_hidden(args.device)

            losses = []
            enc_optimizer.zero_grad()
            binarizer_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            residual = sample_x
            for i in range(train_params['iterations']):
                # print('input:', residual.shape)

                x = encoder(residual)
                x = binarizer(x)
                #print(f"code shape: {x.shape}")
                # nbytes = x.detach().numpy().astype(np.bool).nbytes
                # print('\ncompressed:', x.shape, n_bytes)
                # print()
                output = decoder(x)
                # print('output:', output.shape)

                residual = sample_x - output
                losses.append(residual.abs().mean())

            loss = sum(losses) / train_params['iterations']
            epoch_loss += loss.item()
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            binarizer_optimizer.step()

            if batch_idx % log_interval == 0:
                idx = epoch * int(len(train_loader.dataset) / batch_size) + batch_idx
                writer.add_scalar('loss', loss.item(), idx)
                writer.add_image('input_img', sample_x[0], idx)
                writer.add_image('recon_img', output[0], idx)
                curr_loss = 0

            if batch_idx % val_interval == 0 and train_params['validate']:
                val_loss = 0
                for batch_idx, (sample_x, sample_y) in enumerate(val_loader):

                    sample_x = sample_x.to(args.device)
                    sample_y = sample_y.to(args.device)

                    encoder.init_hidden(args.device)
                    decoder.init_hidden(args.device)

                    x = encoder(sample_x)
                    x = binarizer(x)
                    output = decoder(x)

                    val_loss += l1loss(output, sample_x).item()
                writer.add_scalar('val_loss', val_loss / len(val_loader), idx)
                writer.flush()

                if best_loss > val_loss:
                    best_loss = val_loss
                    save_models(args,
                        encoder,
                        binarizer,
                        decoder,
                        epoch,
                        enc_optimizer,
                        dec_optimizer,
                        binarizer_optimizer,
                        loss)
                    print('Improved: current best_loss on val:{}'.format(best_loss))
                    patience = full_patience
                else:
                    patience -= 1
                    print('patience', patience)
                    if patience == 0:
                        print('Early Stopped: Best L1 loss on val:{}'.format(best_loss))
                        writer.close()
                        return
        print(f"epoch loss: {epoch_loss}")
        writer.add_scalar('epoch loss', epoch_loss, epoch)
        enc_scheduler.step()
        dec_scheduler.step()
        binarizer_scheduler.step()

    print('Finished: Best L1 loss on val:{}'.format(best_loss))
    writer.close()
