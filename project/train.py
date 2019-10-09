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


def inference(args, encoder, binarizer, decoder, sample_x):
    sample_x = torch.unsqueeze(sample_x, 0)
    encoder_h1, encoder_h2, encoder_h3, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = \
        get_hidden_layers(args, sample_x)
    codes = []
    residual = sample_x
    for i in range(args.iterations):
        x, encoder_h1, encoder_h2, encoder_h3 = encoder(
            residual, encoder_h1, encoder_h2, encoder_h3)
        code = binarizer(x)
        output, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = decoder(
            code, decoder_h1, decoder_h2, decoder_h3, decoder_h4)
        residual = residual - output
        codes.append(code.data.cpu().numpy())

    codes = (np.stack(codes).astype(np.int8) + 1) // 2
    shape = codes.shape
    export = np.packbits(codes.reshape(-1))
    print(f"nbytes: {export.nbytes}")
    # np.savez_compressed(args.output, shape=codes.shape, codes=export)

    encoder_h1, encoder_h2, encoder_h3, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = \
        get_hidden_layers(args, sample_x)
    codes = np.unpackbits(export).astype(np.float32)
    codes = np.reshape(codes, shape)
    codes = codes * 2 - 1
    codes = torch.Tensor(codes).to(args.device)
    reconst_img = torch.zeros(1, 3, 256, 256)
    for iters in range(args.iterations):
        output, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = decoder(
            codes[iters], decoder_h1, decoder_h2, decoder_h3, decoder_h4)
        reconst_img = reconst_img + output.data.cpu()
    reconst_img = reconst_img.squeeze(0)
    return reconst_img

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
                x, encoder_h1, encoder_h2, encoder_h3 = encoder(
                    residual, encoder_h1, encoder_h2, encoder_h3)
                x = binarizer(x)
                output, decoder_h1, decoder_h2, decoder_h3, decoder_h4 = decoder(
                    x, decoder_h1, decoder_h2, decoder_h3, decoder_h4)

                if args.additive:
                    residual = residual - output
                else:
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
                if args.additive:
                    writer.add_image(
                        'recon_img',
                        inference(args, encoder, binarizer, decoder, sample_x[0]), idx)
                else:
                    writer.add_image('recon_img', output[0], idx)
                #writer.add_image('recon_img', img_normalize(output[0]), idx)
                curr_loss = 0

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
