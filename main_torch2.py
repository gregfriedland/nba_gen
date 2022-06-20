""" 1D toy GAN example with torch """
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot


# first half of array is positive + second half is negative
def create_real_examples(example_size: int, batch_size: int) -> (torch.Tensor, torch.Tensor):
    values = torch.rand((batch_size, example_size))
    midpoint = example_size // 2
    values[:, :midpoint] = np.abs(values[:, :midpoint])
    values[:, midpoint:] = -np.abs(values[:, midpoint:])
    return values, torch.ones(batch_size)


class Generator(nn.Module):
    def __init__(self, latent_size: int, output_size: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(latent_size), int(output_size))

    def forward(self, x):
        return self.dense_layer(x)


class Discriminator(nn.Module):
    def __init__(self, example_size: int):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(example_size), 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return torch.squeeze(self.activation(self.dense(x)))


def accuracy(pred_labels, true_labels):
    return (pred_labels.round() == true_labels).sum() / len(pred_labels)


def train(latent_size: int = 16, example_size: int = 2, batch_size: int = 512, training_steps: int = 8000):
    # Models
    generator = Generator(latent_size, example_size)
    discriminator = Discriminator(example_size)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # loss
    loss = nn.BCELoss()

    metrics = defaultdict(lambda: [])
    for step_num in range(training_steps):
        ### train discriminator
        discriminator_optimizer.zero_grad()

        # real labels==1
        real_examples, real_labels = create_real_examples(example_size, batch_size)
        pred_real_labels = discriminator(real_examples)
        d_real_loss = loss(pred_real_labels, real_labels)
        d_real_acc = accuracy(pred_real_labels, real_labels)

        # fake labels==0
        fake_examples, fake_labels = generate_fake_examples(batch_size, generator, latent_size)
        pred_fake_labels = discriminator(fake_examples)
        d_fake_loss = loss(pred_fake_labels, fake_labels)
        d_fake_acc = accuracy(pred_fake_labels, fake_labels)

        disc_loss = (d_real_loss + d_fake_loss) / 2
        disc_loss.backward()
        discriminator_optimizer.step()

        ### train generator
        generator_optimizer.zero_grad()

        gen_examples, _ = generate_fake_examples(batch_size, generator, latent_size)
        gen_labels = torch.ones(batch_size)
        pred_gen_labels = discriminator(gen_examples)
        gen_loss = loss(pred_gen_labels, gen_labels)

        gen_loss.backward()
        generator_optimizer.step()

        half_example = example_size // 2
        gen_examples_valid = np.array([(gen_examples[i, half_example] >= 0).all() and
                                       (gen_examples[i, half_example:] <= 0).all()
                              for i in range(batch_size)])

        print(
            f"{step_num:5d} "
            f"d_real_loss={d_real_loss:.3f} "
            f"d_fake_loss={d_fake_loss:.3f} "
            f"gen_loss={gen_loss:0.3f} "
            f"d_real_acc={d_real_acc:0.3f} "
            f"d_fake_acc={d_fake_acc:0.3f} "
            f"gen_pct_valid={gen_examples_valid.mean():0.3f}"
        )
        for var in ["d_real_loss", "d_fake_loss", "gen_loss", "d_real_acc", "d_fake_acc"]:
            metrics[var].append(locals()[var].detach().item())

    plot_history(metrics)


def plot_history(metrics):
    # plot loss
    pyplot.subplot(2, 1, 1)
    for metric in ["d_real_loss", "d_fake_loss", "gen_loss"]:
        pyplot.plot(metrics[metric], label=metric)
    pyplot.legend()

    # plot discriminator accuracy
    pyplot.subplot(2, 1, 2)
    for metric in ["d_real_acc", "d_fake_acc"]:
        pyplot.plot(metrics[metric], label=metric)
    pyplot.legend()

    # save plot to file
    pyplot.savefig('loss_acc.png')
    pyplot.close()

#
# def train_discriminator(batch_size, discriminator, discriminator_optimizer, example_size, loss):
#     discriminator_optimizer.zero_grad()
#
#     # Create real examples from the desired distribution
#     # true_examples = create_real_examples(example_size=example_size, batch_size=batch_size)
#
#     # Train the discriminator on the true/generated data
#     # true_discriminator_out = discriminator(true_examples)
#     # true_discriminator_loss = loss(true_discriminator_out, true_labels)
#     # true_disc_accuracy = (1 == true_discriminator_out.round()).sum() / len(true_discriminator_out)
#
#     # # add .detach() here think about this
#     # fake_disc_out = discriminator(generated_data.detach())
#     # fake_disc_accuracy = (0 == fake_disc_out.round()).sum() / len(true_discriminator_out)
#     # fake_discriminator_loss = loss(fake_disc_out, torch.zeros(batch_size))
#     # discriminator_loss = (true_discriminator_loss + fake_discriminator_loss) / 2
#     discriminator_loss.backward()
#     discriminator_optimizer.step()
#     return fake_disc_accuracy, fake_disc_out, fake_discriminator_loss, true_disc_accuracy, true_discriminator_loss

#
# def train_generator(batch_size, discriminator, generator, generator_optimizer, latent_size, loss):
#     generator_optimizer.zero_grad()
#
#     fake_examples = generate_fake_examples(batch_size, generator, latent_size)
#
#     # does the discriminator think the fake examples look real?
#     fake_validity = discriminator(fake_examples)
#
#     # teach the generator to to make fake examples look real
#     generator_loss = loss(fake_validity, torch.ones(fake_validity.shape))
#     generator_loss.backward()
#     generator_optimizer.step()
#
#     return generator_loss


def generate_fake_examples(batch_size, generator, latent_size):
    latent_points = create_latent_points(batch_size, latent_size)
    return generator(latent_points), torch.zeros(batch_size)


def create_latent_points(batch_size, latent_size):
    latent_values = torch.rand(size=(batch_size, latent_size))
    return latent_values


if __name__ == "__main__":
    train()
