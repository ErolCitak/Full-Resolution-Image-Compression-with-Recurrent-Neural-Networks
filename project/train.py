import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as LS
from PIL import Image
from torchvision import transforms
from tensorboardX import SummaryWriter

from models import Encoder, Decoder, Binarizer


def img_normalize(imgs):
    return (imgs+1.0)/2

def get_hidden_layers(args, sample_x):
    encoder_h1 = (
        torch.zeros(sample_x.size(0), 256, 64, 64).to(args.device),
        torch.zeros(sample_x.size(0), 256, 64, 64).to(args.device)
    )
    encoder_h2 = (
        torch.zeros(sample_x.size(0), 512, 32, 32).to(args.device),
        torch.zeros(sample_x.size(0), 512, 32, 32).to(args.device))
    encoder_h3 = (
        torch.zeros(sample_x.size(0), 512, 16, 16).to(args.device),
        torch.zeros(sample_x.size(0), 512, 16, 16).to(args.device)
    )

    decoder_h1 = (
        torch.zeros(sample_x.size(0), 512, 16, 16).to(args.device),
        torch.zeros(sample_x.size(0), 512, 16, 16).to(args.device)
    )
    decoder_h2 = (
        torch.zeros(sample_x.size(0), 512, 32, 32).to(args.device),
        torch.zeros(sample_x.size(0), 512, 32, 32).to(args.device)
    )
    decoder_h3 = (
        torch.zeros(sample_x.size(0), 256, 64, 64).to(args.device),
        torch.zeros(sample_x.size(0), 256, 64, 64).to(args.device)
    )
    decoder_h4 = (
        torch.zeros(sample_x.size(0), 128, 128, 128).to(args.device),
        torch.zeros(sample_x.size(0), 128, 128, 128).to(args.device)
    )

    return encoder_h1, encoder_h2, encoder_h3, decoder_h1, decoder_h2, decoder_h3, decoder_h4

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
    encoder = Encoder().to(args.device)
    binarizer = Binarizer(args.stochastic).to(args.device)
    decoder = Decoder().to(args.device)

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
            #print(f"batch_idx: {batch_idx}")
            sample_x = sample_x.to(args.device)
            sample_y = sample_y.to(args.device)

            encoder_h1, encoder_h2, encoder_h3, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = get_hidden_layers(args, sample_x)

            losses = []
            #losses = 0
            enc_optimizer.zero_grad()
            binarizer_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            residual = sample_x
            for i in range(train_params['iterations']):
                # print('input:', residual.shape)
                x, encoder_h1, encoder_h2, encoder_h3 = encoder(
                    residual, encoder_h1, encoder_h2, encoder_h3)
                x = binarizer(x)
                # nbytes = x.detach().numpy().astype(np.bool).nbytes
                # print('\ncompressed:', x.shape, n_bytes)
                # print()
                output, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = decoder(
                    x, decoder_h1, decoder_h2, decoder_h3, decoder_h4)
                # print('output:', output.shape)

                residual = sample_x - output
                #residual = residual - output
                #losses += residual.abs().mean()
                losses.append(residual.abs().mean())

            #loss = losses/train_params['iterations']
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
                #writer.add_image('recon_img', img_normalize(output[0]), idx)
                curr_loss = 0

            #if batch_idx % val_interval == 0 and batch_idx != 0:
            if batch_idx % val_interval == 0 and train_params['validate']:
                val_loss = 0
                for batch_idx, (sample_x, sample_y) in enumerate(val_loader):
                    sample_x = sample_x.to(args.device)
                    sample_y = sample_y.to(args.device)
                    encoder_h1, encoder_h2, encoder_h3, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = get_hidden_layers(args, sample_x)
                    x, encoder_h1, encoder_h2, encoder_h3 = encoder(
                        sample_x, encoder_h1, encoder_h2, encoder_h3)
                    x = binarizer(x)
                    output, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = decoder(
                        x, decoder_h1, decoder_h2, decoder_h3, decoder_h4)

                    val_loss += l1loss(output, sample_x).item()
                writer.add_scalar('val_loss', val_loss / len(val_loader), idx)
                writer.flush()

                if best_loss > val_loss:
                    best_loss = val_loss
                    best_encoder = copy.deepcopy(encoder)
                    best_binarizer = copy.deepcopy(binarizer)
                    best_decoder = copy.deepcopy(decoder)
                    save_models(args,
                        encoder,
                        binarizer,
                        decoder,
                        epoch,
                        enc_optimizer,
                        dec_optimizer,
                        binarizer_optimizer,
                        loss)
                    #save_models(args, encoder, binarizer, decoder)
                    print('Improved: current best_loss on val:{}'.format(best_loss))
                    patience = full_patience
                else:
                    patience -= 1
                    print('patience', patience)
                    if patience == 0:
                        save_models(args,
                            best_encoder,
                            best_binarizer,
                            best_decoder,
                            epoch,
                            enc_optimizer,
                            dec_optimizer,
                            binarizer_optimizer,
                            loss)
                        #save_models(args, encoder, binarizer, decoder)
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
