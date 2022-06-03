# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
import argparse
from typing import Dict

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from data import ON_OFFENSE, ON_DEFENSE


class Generator(nn.Module):
    def __init__(self, example_size: int, condition_size: int, latent_dim: int):
        super(Generator, self).__init__()
        self.example_size = example_size
        self.condition_size = condition_size
        self.latent_dim = latent_dim

        # self.condition_embedding = nn.Embedding(condition_size, condition_size)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + condition_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, example_size),
            nn.Tanh()
        )

    def forward(self, latent_vector: torch.FloatTensor, conditions: torch.LongTensor):
        # g_in = torch.cat((self.condition_embedding(conditions), latent_vector), -1)
        g_in = torch.cat((conditions, latent_vector), -1)
        return self.model(g_in)

    def fake_conditions(self, batch_size: int) -> torch.LongTensor:
        """ Generate a fake batch of conditions"""
        assert self.condition_size % 2 == 0  # must be event
        all_conditions = []
        for i in range(batch_size):
            # pick two different entities (teams) to be on offense/defense
            # loop over batch_size for now b/c don't know of a np native way to sample
            # without replacement on one dim only
            entities = np.random.choice(self.condition_size // 2, size=2, replace=False)
            entities[1] += self.condition_size // 2
            conditions = torch.LongTensor(np.zeros(self.condition_size))
            conditions[entities] = 1
            all_conditions.append(conditions)
        return torch.stack(all_conditions)

    def fake_latents(self, batch_size: int) -> torch.FloatTensor:
        return torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim)))


class Discriminator(nn.Module):
    def __init__(self, example_size: int, condition_size: int):
        super(Discriminator, self).__init__()
        self.example_size = example_size
        self.condition_size = condition_size

        # self.condition_embedding = nn.Embedding(condition_size, condition_size)

        self.model = nn.Sequential(
            nn.Linear(condition_size + example_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, stats: torch.FloatTensor, conditions: torch.LongTensor):
        # Concatenate label embedding and image to produce input
        # d_in = torch.cat((stats, self.condition_embedding(conditions)), -1)
        d_in = torch.cat((stats, conditions), -1)
        return self.model(d_in)


class PbpDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, stat_col_prefix: str):
        stat_tensors = {}
        condition_tensors = {}
        for col in df.columns:
            if col.startswith(stat_col_prefix):
                tensor = torch.tensor(df[col].values, dtype=torch.float32)
                stat_tensors[col] = nn.functional.normalize(tensor, dim=0)
            else:
                on_offense = df[col] == ON_OFFENSE
                on_defense = df[col] == ON_DEFENSE
                condition_tensors[col + "_off"] = torch.tensor(on_offense, dtype=torch.long)
                condition_tensors[col + "_def"] = torch.tensor(on_defense, dtype=torch.long)
        self.stats = torch.stack(list(stat_tensors.values()), dim=1)
        self.conditions = torch.stack(list(condition_tensors.values()), dim=1)

    def __len__(self):
        return len(self.stats)

    def __getitem__(self, index):
        return self.stats[index, :], self.conditions[index, :]


def eval(generator: Generator, optimizer_G, opt, device: str, gen_checkpoint_file: str):
    gen_checkpoint = torch.load(gen_checkpoint_file)
    generator.load_state_dict(gen_checkpoint['model_state_dict'])
    optimizer_G.load_state_dict(gen_checkpoint['optimizer_state_dict'])
    generator.eval()

    for i in range(10000):
        fake_latent = generator.fake_latents(opt.batch_size).to(device)
        fake_conditions = generator.fake_conditions(opt.batch_size).to(device)
        gen_events = generator(fake_latent, fake_conditions)
        pass


def train(adversarial_loss, dataloader, discriminator: Discriminator, generator: Generator, opt,
          optimizer_D: torch.optim.Adam, optimizer_G: torch.optim.Adam, device: str, checkpoint_files: Dict[str, str]):
    for epoch in range(opt.n_epochs):
        for i, (stats, conditions) in enumerate(dataloader):
            # Adversarial ground truths
            real_truth = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(1.0).to(device), requires_grad=False)
            fake_truth = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(0.0).to(device), requires_grad=False)

            # Configure input
            # stats = Variable(stats.type(FloatTensor))
            # conditions = Variable(conditions.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            fake_latent = Variable(generator.fake_latents(opt.batch_size))
            fake_conditions = Variable(generator.fake_conditions(opt.batch_size))

            # Generate a fake batch
            fake_stats = generator(fake_latent, fake_conditions)

            # Loss measures generator's ability to generate real-looking stats from fake inputs
            validity = discriminator(fake_stats, fake_conditions)
            g_loss = adversarial_loss(validity, real_truth)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real examples
            validity_real = discriminator(Variable(stats), Variable(conditions))
            d_real_loss = adversarial_loss(validity_real, real_truth[:len(validity_real), :])

            # Loss for fake examples
            validity_fake = discriminator(fake_stats.detach(), fake_conditions)
            d_fake_loss = adversarial_loss(validity_fake, fake_truth)

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
    parser.add_argument("--n_epochs", type=int, default=25, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2048, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    # parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
    parser.add_argument("mode", type=str, choices=["train", "eval"], help="train or eval mode")
    parser.add_argument("data_file", type=argparse.FileType('r'), help="data csv file")
    opt = parser.parse_args()
    print(opt)

    # Configure data loader
    df = pd.read_csv(opt.data_file)
    dataset = PbpDataset(df, "stat.")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss functions
    adversarial_loss = torch.nn.MSELoss().to(device)

    # Initialize generator and discriminator
    stats_size = dataset.stats.shape[1]
    conditions_size = dataset.conditions.shape[1]
    generator = Generator(stats_size, conditions_size, opt.latent_dim).to(device)
    discriminator = Discriminator(stats_size, conditions_size).to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    checkpoint_files = {
        "generator": opt.data_file.name.replace(".csv", ".gen.pt"),
        "discriminator": opt.data_file.name.replace(".csv", ".disc.pt")
    }

    if opt.mode == "train":
        train(adversarial_loss, dataloader, discriminator, generator, opt, optimizer_D, optimizer_G, device, checkpoint_files)
    else:
        eval(generator, optimizer_G, opt, device, checkpoint_files["generator"])


if __name__ == "__main__":
    main()
