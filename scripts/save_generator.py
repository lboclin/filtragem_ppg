import torch
from gan_generator import PPGGenerator

# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/generator.pth'

if __name__ == "__main__":
    # Inicializa o Gerador
    generator = PPGGenerator().to(DEVICE)

    # Carrega o modelo treinado da memória (depois de treino)
    generator.load_state_dict(torch.load('C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/scripts/generator_trained.pth', map_location=DEVICE))

    # Salva o Gerador treinado
    torch.save(generator.state_dict(), MODEL_SAVE_PATH)

    print(f"✅ Gerador salvo em: {MODEL_SAVE_PATH}")
