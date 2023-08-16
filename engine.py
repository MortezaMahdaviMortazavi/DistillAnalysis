import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from dataset import CustomCIFARDataset
from sklearn.metrics import accuracy_score, f1_score
from loss import DistillLoss
from StudentModels import Cifar10Classifier
from tqdm import tqdm


if __name__ == "__main__":
    ################################# configs ###############################################################################
    train_root_dir = 'cifar10/train/'
    test_root_dir = 'cifar10/test/'

    image_size = (128,128)
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_name = 'resnet18'
    assert teacher_name in ['resnet18','resnet34','resnet50','resnet101','resnet152','vit','swin']

    # criterion = DistillLoss(temperature=1)
    criterion = nn.CrossEntropyLoss()
    student_model = Cifar10Classifier().to(device)
    student_model.load_state_dict(torch.load('StudentCheckpoints/CifarCNNClassifier.pt'))
    teacher_model = ...
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

    

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])  # Normalize to [-1, 1]
    ])

    # Define test transform without augmentation
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomCIFARDataset(root_dir=train_root_dir,transform=train_transform)
    test_dataset = CustomCIFARDataset(root_dir=test_root_dir,transform=test_transform)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    #####################################################################################################################


    ##### Training ####
    # teacher_model.eval()

    for epoch in range(num_epochs):
        student_model.train()  # Set the student_model to training mode
        total_loss = 0.0

        for images, labels in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = student_model(images)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}')

        # Evaluation loop
        student_model.eval()  # Set the student_model to evaluation mode
        all_labels = []
        all_preds = []
        ########################################### Testing ###############################################
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.4f}, F-score: {f1:.4f}')

    torch.save(student_model.state_dict(),'StudentCheckpoints/CifarCNNClassifier.pt')
    print("Training finished!")
    #######################################################################################################