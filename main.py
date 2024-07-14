import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import spacy

torch.manual_seed(42)
np.random.seed(42)

class ConditionalGenerator(nn.Module):
    def __init__(self, nz, ngf, nc, text_embedding_dim):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Linear(text_embedding_dim, nz)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz * 2, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        text_embedding = self.label_embedding(text_embedding)
        text_embedding = text_embedding.unsqueeze(-1).unsqueeze(-1) 
        combined_input = torch.cat((noise, text_embedding), 1)
        return self.main(combined_input)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class CustomDataset(Dataset):
    def __init__(self, root_image_dir, root_text_dir, transform=None):
        self.root_image_dir = root_image_dir
        self.root_text_dir = root_text_dir
        self.image_files = os.listdir(root_image_dir)
        self.text_files = os.listdir(root_text_dir)
        self.transform = transform

    def __len__(self):
        return min(len(self.image_files), len(self.text_files))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        text_name = os.path.join(self.root_text_dir, self.text_files[idx])
        with open(text_name, 'r') as file:
            description = file.read()

        return image, description


nlp = spacy.load("en_core_web_md")


def encode_sentence(sentence):
    with torch.no_grad():
        tokenized = nlp(sentence)
        word_embeddings = [token.vector for token in tokenized if token.is_alpha]
        if word_embeddings:
            encoded_vector = torch.tensor(np.mean(word_embeddings, axis=0), dtype=torch.float32)
            if encoded_vector.size(0) < nz:
                encoded_vector = torch.cat((encoded_vector, torch.zeros(nz - encoded_vector.size(0))))
            elif encoded_vector.size(0) > nz:
                encoded_vector = encoded_vector[:nz]
            return encoded_vector
        else:
            return torch.zeros(nz, dtype=torch.float32)


# Hyperparameters and setup
batch_size = 124
image_size = 64
nc = 3
nz = 128
ngf = 64
ndf = 64
num_epochs = 300
text_embedding_dim = nz  


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


dataset = CustomDataset(root_image_dir='path',
                        root_text_dir='path',
                        transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = ConditionalGenerator(nz, ngf, nc, text_embedding_dim).to(device)
netD = Discriminator(nc, ndf).to(device)




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G_losses = []
D_losses = []
netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))


# Training Loop
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        if data is None:
            continue
        images, descriptions = data
        b_size = images.size(0)
        real_images = images.to(device)
        
        encoded_texts = [encode_sentence(description).to(device) for description in descriptions]
        encoded_texts = torch.stack(encoded_texts)

        netD.zero_grad()
        label = torch.full((b_size,), 1, dtype=torch.float, device=device)
        output = netD(real_images).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = netG(noise, encoded_texts)
        label.fill_(0)
        output = netD(fake_images.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(1)
        output = netD(fake_images).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item()} Loss_G: {errG.item()}')
        
        
                
        G_losses.append(errG.item())
        D_losses.append(errD.item())

def generate_image_from_description(description):
    netG.eval()
    description_vector = encode_sentence(description).to(device)
    noise = torch.randn(1, nz, 1, 1, device=device)
    with torch.no_grad():
        fake_image = netG(noise, description_vector.unsqueeze(0)).squeeze(0).cpu()
    
    fake_image = np.transpose(fake_image.numpy(), (1, 2, 0))
    
    plt.imshow((fake_image + 1) / 2)  # Normalize the image to [0, 1] for proper visualization
    plt.axis('off')
    plt.show()


plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Example usage to generate images from descriptions
test_cases = [
    "A blue sedan car",
    "A black suv car ",
    "A white sedan car",
    "A yellow sedan car",
    "A green suv car"
    "A blue sports car",
    "A white SUV car",
    "A red truck",
    "A red sedan car",
    "A black sedan car"
    "A green suv car",
    "A silver sedan car",
    "A white sedan car",
    "A gray sedan car",
    "A green suv car"
]

for description in test_cases:
    generate_image_from_description(description)

