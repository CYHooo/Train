import torch
from torch import nn


class training():

    def __init__(self) -> None:
        pass


    def train(
            model: nn.Module,
            train_loader: torch.Tensor,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.Module,
            num_epoch: int,
            device: torch.device
            ):
        
        ## move to GPU
        model = model.to(device)
        
        for epoch in range(num_epoch):
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 2 == 0:
                    print(f"Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}")

    def val(
            model: nn.Module,
            test_loader: torch.Tensor,
            device: torch.device
            ):
        predicts = []
        ## move model to GPU
        model = model.to(device)

        model.eval()  # switch to evaluation mode

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                outputs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predicts.append(predicted)
            print(f"Test Accuracy of the model on the test images: {100 * correct / total}%")



    def test(
            model: nn.Module,
            test_loader: torch.Tensor,
            device: torch.device
            ):
        predicts = []
        ## move model to GPU
        model = model.to(device)

        model.eval()  # switch to evaluation mode

        with torch.no_grad():
            for images in test_loader:

                images = images[0].to(device)


                outputs = model(images)
                outputs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                predicts.append(predicted)

        return predicts