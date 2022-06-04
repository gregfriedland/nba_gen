# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
import argparse
import logging
import os
from collections.abc import Sequence
from copy import copy, deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from data import ON_OFFENSE, ON_DEFENSE, NOT_PLAYING


class Entities:
    def __init__(self, entities: List[str], num_entities_per_type: int):
        self.entities = entities
        self.num_entities = len(entities)
        self.num_entities_per_type = num_entities_per_type
        self.types = (ON_OFFENSE, ON_DEFENSE)
        self.entities_by_type = {
            ON_OFFENSE: [],
            ON_DEFENSE: []
        }

    def is_full(self, type: str):
        assert len(self.entities_by_type[type]) <= self.num_entities_per_type, f"Conditions {type} is overfull"
        return len(self.entities_by_type[type]) == self.num_entities_per_type

    def add(self, type: str, entities: List[str]):
        self.entities_by_type[type].extend(entities)
        return self.is_full(type)

    def fill(self):
        new_obj = deepcopy(self)
        for type in self.types:
            while len(new_obj.entities_by_type[type]) < self.num_entities_per_type:
                new_entity = np.random.choice(self.entities, 1)
                if new_entity not in new_obj.entities_by_type[type]:
                    new_obj.entities_by_type[type].append(new_entity)
        return new_obj

    def build_condition_tensor(self):
        new_obj = self.fill()

        tensors = []
        for type in new_obj.types:
            tensor = torch.LongTensor(np.zeros(self.num_entities))
            for entity in new_obj.entities_by_type[type]:
                tensor[self.entities.index(entity)] = 1  # TODO this could be faster
            tensors.append(tensor)
        return torch.cat(tensors, -1)


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
            *block(latent_dim + condition_size, 1024, normalize=False),
            # *block(128, 256),
            # *block(256, 512),
            *block(1024, 1024),
            *block(1024, 1024),
            nn.Linear(1024, example_size),
            nn.Tanh()
        )

    def forward(self, latent_vector: torch.FloatTensor, conditions: torch.LongTensor):
        # g_in = torch.cat((self.condition_embedding(conditions), latent_vector), -1)
        g_in = latent_vector # torch.cat((conditions, latent_vector), -1)
        return self.model(g_in)

    def fake_conditions(self, batch_size: int, condition_entities: Entities) -> Tensor:
        """ Generate a fake batch of conditions"""
        all_conditions = []
        for i in range(batch_size):
            all_conditions.append(condition_entities.build_condition_tensor())
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
            # nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            # nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, stats: torch.FloatTensor, conditions: torch.LongTensor):
        # Concatenate label embedding and image txo produce input
        # d_in = torch.cat((stats, self.condition_embedding(conditions)), -1)
        d_in = stats # torch.cat((stats, conditions), -1)
        return self.model(d_in)


class PbpDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, stat_col_prefix: str):
        df = df[df.NOP == ON_OFFENSE].iloc[:2048] ## DEBUG
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
        self.df = df
        self.stat_col_prefix = stat_col_prefix

    def __len__(self):
        return len(self.stats)

    def __getitem__(self, index):
        return self.stats[index, :], self.conditions[index, :]

    @property
    def teams_based(self):
        return len(self.entity_names) == 30

    @property
    def num_entities_per_type(self):
        return 1 if self.teams_based else 5

    @property
    def entity_names(self):
        return [c for c in self.df.columns if not c.startswith(self.stat_col_prefix)]

    @property
    def stat_names(self):
        return sorted(set(self.df.columns).difference(self.entity_names))

    def df_from_tensor(self, tensor: Tensor, has_entities: bool, has_stats: bool):
        num_conditions = 2 * len(self.entity_names)
        dfs = []
        if has_entities:
            dfs.append(pd.DataFrame(tensor[:, :num_conditions].detach().numpy(), columns=2*self.entity_names))
            tensor = tensor[:, num_conditions:]
        if has_stats:
            dfs.append(pd.DataFrame(tensor[:, :len(self.stat_names)].detach().numpy(), columns=self.stat_names))
        return pd.concat(dfs, axis=1)


def calc_summary_stats(df: pd.DataFrame, entity_names: List[str], stat_names: List[str]):
    all_stats = []
    for entity in entity_names:
        entity_df = df[df[entity] != NOT_PLAYING]
        stats = {"entity": entity}
        stats.update({f"{stat}_mean": entity_df[stat].mean() for stat in stat_names})
        # stats.update({f"{stat}_std": entity_df[stat].std() for stat in stat_names})
        all_stats.append(stats)

    return pd.DataFrame(all_stats)


def eval(dataset: PbpDataset, generator: Generator, optimizer_G, opt, device: str, gen_checkpoint_file: str):
    # calc summary stats from real training data
    df = dataset.df
    real_normalized_df = (df - df.mean(axis=1)) / df.std(axis=1)
    real_stats_df = calc_summary_stats(real_normalized_df, dataset.entity_names, dataset.stat_names)

    real_stats_df.to_csv(gen_checkpoint_file.replace(".pt", ".real.stats.csv"))

    gen_checkpoint = torch.load(gen_checkpoint_file)
    generator.load_state_dict(gen_checkpoint['model_state_dict'])
    optimizer_G.load_state_dict(gen_checkpoint['optimizer_state_dict'])
    generator.eval()

    all_fake_events = []
    num_entities = len(dataset.entity_names)
    for entity in dataset.entity_names:  # TODO DEBUG
        logging.debug(f"Generating events for {entity}")
        entities = Entities(dataset.entity_names, dataset.num_entities_per_type)
        entities.add(ON_OFFENSE, [entity])  # TODO defense

        num_batches = len(dataset) // (num_entities * opt.batch_size)
        for i in range(num_batches):
            fake_latent = generator.fake_latents(opt.batch_size).to(device)
            fake_conditions = generator.fake_conditions(opt.batch_size, entities).to(device)
            fake_stats = generator(fake_latent, fake_conditions).detach()
            all_fake_events.append(torch.cat([fake_conditions[:, :len(dataset.entity_names)], fake_stats], dim=-1))

    fake_events_df = pd.DataFrame(data=torch.cat(all_fake_events, 0), columns=dataset.entity_names + dataset.stat_names)
    fake_stats_df = calc_summary_stats(fake_events_df, dataset.entity_names, dataset.stat_names)
    fake_stats_df.to_csv(gen_checkpoint_file.replace(".pt", ".fake.stats.csv"))


def train(adversarial_loss, dataset: PbpDataset, discriminator: Discriminator, generator: Generator, opt,
          optimizer_D: torch.optim.Adam, optimizer_G: torch.optim.Adam, device: str, checkpoint_files: Dict[str, str]):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    for epoch in range(opt.n_epochs):
        for i, (stats, conditions) in enumerate(dataloader):
            # Adversarial ground truths
            real_truth = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(1.0).to(device), requires_grad=False)
            fake_truth = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(0.0).to(device), requires_grad=False)

            #  Train Generator
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            fake_latent = Variable(generator.fake_latents(opt.batch_size))
            # fake_conditions = Variable(generator.fake_conditions(opt.batch_size,
            #                                                      Entities(dataset.entity_names, dataset.num_entities_per_type)))

            # Generate a fake batch
            fake_stats = generator(fake_latent, conditions)

            if os.getenv("DEBUG", None):
                fake_df = dataset.df_from_tensor(torch.cat([conditions, fake_stats], dim=1), True, True)
                logging.debug(f"fake_df:\n{fake_df.head()}")

            # Loss measures generator's ability to generate real-looking stats from fake inputs
            validity = discriminator(fake_stats, conditions)
            g_loss = adversarial_loss(validity, real_truth)

            g_loss.backward()
            optimizer_G.step()

            g_loss_updated = adversarial_loss(discriminator(generator(fake_latent, conditions), conditions), real_truth)
            logging.debug(f"g_loss={g_loss} g_loss_update={g_loss_updated}")

            #  Train Discriminator
            optimizer_D.zero_grad()

            # Loss for real examples
            validity_real = discriminator(Variable(stats), Variable(conditions))
            d_real_loss = adversarial_loss(validity_real, real_truth[:len(validity_real), :])

            # Loss for fake examples
            validity_fake = discriminator(fake_stats.detach(), conditions)
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
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
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

    logging.basicConfig(level=logging.DEBUG)

    # Configure data loader
    df = pd.read_csv(opt.data_file)
    dataset = PbpDataset(df, "stat.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss functions
    adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)

    # Initialize generator and discriminator
    stats_size = dataset.stats.shape[1]
    conditions_size = dataset.conditions.shape[1]
    # generator = Generator(stats_size, conditions_size, opt.latent_dim).to(device)
    # discriminator = Discriminator(stats_size, conditions_size).to(device)
    generator = Generator(stats_size, 0, opt.latent_dim).to(device)
    discriminator = Discriminator(stats_size, 0).to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    checkpoint_files = {
        "generator": opt.data_file.name.replace(".csv", ".gen.pt"),
        "discriminator": opt.data_file.name.replace(".csv", ".disc.pt")
    }

    if opt.mode == "train":
        train(adversarial_loss, dataset, discriminator, generator, opt, optimizer_D, optimizer_G, device, checkpoint_files)
    else:
        eval(dataset, generator, optimizer_G, opt, device, checkpoint_files["generator"])


if __name__ == "__main__":
    main()
