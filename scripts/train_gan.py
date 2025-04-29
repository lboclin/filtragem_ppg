import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gan_generator import PPGGenerator
from gan_discriminator import PPGDiscriminator

# Configurações
PROCESSED_DIR = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/data/processed/'
NOISY_DIR = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/data/noisy/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 200
BATCH_SIZE = 4
LR = 0.0002

# Função para carregar lote de treino
def load_batch(file_list, batch_size, idx):
    clean_batch = []
    noisy_batch = []
    for i in range(batch_size):
        if idx + i >= len(file_list):
            break
        clean = np.load(os.path.join(PROCESSED_DIR, file_list[idx + i].replace('_noisy', '')))
        noisy = np.load(os.path.join(NOISY_DIR, file_list[idx + i]))
        clean_batch.append(clean)
        noisy_batch.append(noisy)
    
    clean_batch = np.array(clean_batch)
    noisy_batch = np.array(noisy_batch)

    clean_batch = torch.tensor(clean_batch, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    noisy_batch = torch.tensor(noisy_batch, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    return clean_batch, noisy_batch

if __name__ == "__main__":
    # Inicializa modelos
    generator = PPGGenerator().to(DEVICE)
    discriminator = PPGDiscriminator().to(DEVICE)

    # Loss functions
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.L1Loss()

    # Otimizadores
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    # Lista de arquivos
    file_list = [f for f in os.listdir(NOISY_DIR) if f.endswith('.npy')]
    print(f"🔍 {len(file_list)} arquivos encontrados para treinamento.")

    for epoch in range(EPOCHS):
        np.random.shuffle(file_list)
        g_loss_epoch = 0
        d_loss_epoch = 0

        for idx in range(0, len(file_list), BATCH_SIZE):
            real_clean, noisy_input = load_batch(file_list, BATCH_SIZE, idx)

            # ======== Treina Discriminador ========
            optimizer_D.zero_grad()

            # Sinais reais
            valid = torch.ones(real_clean.size(0), 1).to(DEVICE)
            # Sinais falsos
            fake = torch.zeros(real_clean.size(0), 1).to(DEVICE)

            # Avaliação real
            real_pred = discriminator(real_clean)
            real_loss = adversarial_loss(real_pred, valid)

            # Avaliação falsa
            generated_clean = generator(noisy_input)
            fake_pred = discriminator(generated_clean.detach())
            fake_loss = adversarial_loss(fake_pred, fake)

            # Loss total do Discriminador
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # ======== Treina Gerador ========
            optimizer_G.zero_grad()

            generated_clean = generator(noisy_input)
            fake_pred = discriminator(generated_clean)

            # O Gerador quer que o Discriminador acredite que as imagens geradas são reais
            g_adv_loss = adversarial_loss(fake_pred, valid)
            g_rec_loss = reconstruction_loss(generated_clean, real_clean)

            # Loss total do Gerador (reconstrução + adversarial)
            g_loss = g_adv_loss + 10 * g_rec_loss  # 10 é um peso para dar mais força à reconstrução
            g_loss.backward()
            optimizer_G.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

        print(f"[Época {epoch+1}/{EPOCHS}] Loss G: {g_loss_epoch:.4f} | Loss D: {d_loss_epoch:.4f}")

    print("✅ Treinamento da GAN concluído!")

    torch.save(generator.state_dict(), 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/generator.pth')
    print("✅ Gerador salvo como 'generator.pth'")
