# test_torch_hub.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Este script prueba que los modelos definidos en hubconf.py se pueden cargar correctamente.
# No usarlo como test unitario formal, solo para verificar integridad.
# ============================================================
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import hubconf


def main():

    print("Loading standalone cvae")
    cvae = hubconf.cvae_standalone()
    for seq_len in [32, 64, 128]:
        print(f"Loading cvae from hybrid with seq_len: {seq_len}")
        cvae = hubconf.cvae_hybrid(seq_len)
    del cvae

    print("Loading standalone generator")
    generator = hubconf.generator_standalone()
    for seq_len in [32, 64, 128]:
        print(f"Loading generator from hybrid with seq_len: {seq_len}")
        generator = hubconf.generator_hybrid(seq_len)
    del generator

    print("Loading standalone critic")
    critic = hubconf.critic_standalone()
    for seq_len in [32, 64, 128]:
        print(f"Loading critic from hybrid with seq_len: {seq_len}")
        critic = hubconf.critic_hybrid(seq_len)
    del critic


if __name__ == '__main__':
    main()
