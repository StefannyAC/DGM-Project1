# hubconf.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Este archivo permite cargar modelos preentrenados desde PyTorch Hub
# ------------------------------------------------------------
import torch


from cvae_seq2seq import CVAE
from cgan import Generator, Critic


dependencies = ['torch']
DOWNLOAD_URL = "https://github.com/StefannyAC/DGM-Project1/releases/download/weights-1.0/"


def cvae_standalone(pretrained: bool=True) -> torch.nn.Module:
    model = CVAE(
        z_dim=32,
        cond_dim=4,
        seq_len=32,
        ev_embed=16,
        cond_embed=4,
        enc_hid=16,
        dec_hid=16
    )

    if not pretrained:
        return model

    url = f"{DOWNLOAD_URL}/cvae_pretrained_best.pth"
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


def cvae_hybrid(seq_len: int, pretrained: bool=True) -> torch.nn.Module:
    model = CVAE(
        z_dim=32,
        cond_dim=4,
        seq_len=seq_len,
        ev_embed=16,
        cond_embed=4,
        enc_hid=16,
        dec_hid=16
    )

    if not pretrained:
        return model

    if seq_len == 32:
        url = f"{DOWNLOAD_URL}/hybrid_cvae_best_T32.pth"
    elif seq_len == 64:
        url = f"{DOWNLOAD_URL}/hybrid_cvae_best_T64.pth"
    elif seq_len == 128:
        url = f"{DOWNLOAD_URL}/hybrid_cvae_morepoch_best_T128.pth "
    else:
        raise ValueError(f"Invalid sequence length. Try: 32, 64 or 128")

    checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


def generator_standalone(pretrained: bool=True) -> torch.nn.Module:
    model = Generator(
        z_dim=32,
        cond_dim=4,
        seq_len=32,
        hidden_dim=32
    )

    if not pretrained:
        return model

    url = f"{DOWNLOAD_URL}/generator_pretrained_best.pth"
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


def generator_hybrid(seq_len: int, pretrained: bool=True) -> torch.nn.Module:
    model = Generator(
        z_dim=32,
        cond_dim=4,
        seq_len=seq_len,
        hidden_dim=512
    )

    if not pretrained:
        return model

    if seq_len == 32:
        url = f"{DOWNLOAD_URL}/hybrid_G_best_T32.pth"
    elif seq_len == 64:
        url = f"{DOWNLOAD_URL}/hybrid_G_best_T64.pth"
    elif seq_len == 128:
        url = f"{DOWNLOAD_URL}/hybrid_G_morepoch_best_T128.pth"
    else:
        raise ValueError(f"Invalid sequence length. Try: 32, 64 or 128")

    checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


def critic_standalone(pretrained: bool=True) -> torch.nn.Module:
    model = Critic(
        cond_dim=4,
        seq_len=32,
        hidden_dim=32
    )

    if not pretrained:
        return model
    
    url = f"{DOWNLOAD_URL}/critic_pretrained_best.pth"
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


def critic_hybrid(seq_len: int, pretrained: bool=True) -> torch.nn.Module:
    model = Critic(
        cond_dim=4,
        seq_len=seq_len,
        hidden_dim=512,
    )

    if not pretrained:
        return model
    
    if seq_len == 32:
        url = f"{DOWNLOAD_URL}/hybrid_D_best_T32.pth"
    elif seq_len == 64:
        url = f"{DOWNLOAD_URL}/hybrid_D_best_T64.pth"
    elif seq_len == 128:
        url = f"{DOWNLOAD_URL}/hybrid_D_morepoch_best_T128.pth"
    else:
        raise ValueError(f"Invalid sequence length. Try: 32, 64 or 128")

    checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model
