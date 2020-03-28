from network.CNN import CNNClassifier
import torch

if __name__ == '__main__':
    save_path = 'cnn_model.pth'
    checkpoint = torch.load(save_path)
    cnn = CNNClassifier()
    cnn.load_state_dict(checkpoint['model'])
    cnn.eval()
    print(cnn)
