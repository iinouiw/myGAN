import torch

from dataset import HorseZebraDataset
from utils import  load_checkpoint
from torch.utils.data import DataLoader
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_net import Discriminator
from generator_net import Generator

def test_fn( gen_Z, gen_H, loader,):
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            fake_zebra = gen_Z(horse)
            circle_horse=gen_H(fake_zebra)
            circle_zebra=gen_Z(fake_horse)
        if idx >=0:
            prehorse=torch.cat([zebra,fake_horse,circle_zebra],0)
            prezebra=torch.cat([horse,fake_zebra,circle_horse],0)
            save_image(prehorse*0.5+0.5, config.SAVE_DIR+f"/zebra2horse/zebra2horse_{idx}.png")
            save_image(prezebra*0.5+0.5,config.SAVE_DIR +f"/horse2zebra/horse2zebra_{idx}.png")




def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )


    if config.LOAD_MODEL:

        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )

        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    val_dataset = HorseZebraDataset(
       root_horse=config.VAL_DIR+"/horses", root_zebra=config.VAL_DIR+"/zebras", transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    test_fn(gen_Z,gen_H,val_loader)



if __name__ == "__main__":
    main()