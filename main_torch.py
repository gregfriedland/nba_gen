# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from data import PLAYER_ID_PREFIX, ON_OFFENSE, ON_DEFENSE


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, example_shape):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, example_shape[1]),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        return self.model(gen_input)


class Discriminator(nn.Module):
    def __init__(self, n_classes, example_shape):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + example_shape[1], 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img, self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class PbpDataset(torch.utils.data.Dataset):
    def __init__(self, df, label_col):
        tensors = {}
        for col in df.columns:
            if col in ("fg:shooter", "fg:assister", "fg:defender", "fg:blocker"):
                continue
            elif col.startswith(PLAYER_ID_PREFIX):
                on_offense = df[col] == ON_OFFENSE
                on_defense = df[col] == ON_DEFENSE
                tensors[col + "_off"] = torch.tensor(on_offense, dtype=torch.bool)
                tensors[col + "_def"] = torch.tensor(on_defense, dtype=torch.bool)
            elif df[col].dtype == "bool":
                tensors[col] = torch.tensor(df[col].values, dtype=torch.long)
            else:
                tensor = torch.tensor(df[col].values, dtype=torch.float32)
                tensors[col] = nn.functional.normalize(tensor, dim=0)
        self.y = tensors[label_col]
        self.x = torch.stack([t for col, t in tensors.items() if col != label_col], dim=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index, :], self.y[index]


def eval(generator, optimizer_G, opt, gen_checkpoint_file):
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    gen_checkpoint = torch.load(gen_checkpoint_file)
    generator.load_state_dict(gen_checkpoint['model_state_dict'])
    optimizer_G.load_state_dict(gen_checkpoint['optimizer_state_dict'])
    generator.eval()

    for i in range(10000):
        z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, opt.batch_size)))
        gen_events = generator(z, gen_labels)
        pass


def train(adversarial_loss, dataloader, discriminator, generator, opt, optimizer_D, optimizer_G, checkpoint_files):
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    for epoch in range(opt.n_epochs):
        for i, (rows, labels) in enumerate(dataloader):
            batch_size = rows.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real = Variable(rows.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a fake batch
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real examples
            validity_real = discriminator(real, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake examples
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
    torch.save({
        'epoch': opt.n_epochs,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer_G.state_dict(),
    }, checkpoint_files["generator"])
    torch.save({
        'epoch': opt.n_epochs,
        'model_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizer_D.state_dict(),
    }, checkpoint_files["discriminator"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4096, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
    parser.add_argument("mode", type=str, choices=["train", "eval"], help="train or eval mode")
    parser.add_argument("data_file", type=argparse.FileType('r'), help="data csv file")
    opt = parser.parse_args()
    print(opt)

    # Configure data loader
    df = pd.read_csv(opt.data_file)
    dataset = PbpDataset(df, "fg:is_made")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    example_shape = (1, dataset.x.shape[1])
    generator = Generator(opt.n_classes, opt.latent_dim, example_shape)
    discriminator = Discriminator(opt.n_classes, example_shape)

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    checkpoint_files = {
        "generator": opt.data_file.name.replace(".csv", ".gen.pt"),
        "discriminator": opt.data_file.name.replace(".csv", ".disc.pt")
    }

    if opt.mode == "train":
        train(adversarial_loss, dataloader, discriminator, generator, opt, optimizer_D, optimizer_G, checkpoint_files)
    else:
        eval(generator, optimizer_G, opt, checkpoint_files["generator"])


if __name__ == "__main__":
    main()
