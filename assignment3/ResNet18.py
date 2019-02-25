import os
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms.functional import to_pil_image, to_tensor, normalize
import torch
from torch import nn
from dataloaders import load_cifar10, mean, std
from utils import to_cuda, compute_loss_and_accuracy

#import CNN Model
from deepCNN_2 import DeepCNNModel


class ResNetModel(nn.Module):

    def __init__(self,
                 num_classes):
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512*4, num_classes)   # No need to apply softmax ,
                                                        # as this is done in nn. CrossEntropyLoss
        
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters():    # Unfreeze the last fully - connected
            param.requires_grad = True              #layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True               #layers

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=8)
        x = self.model(x)
        return x

def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

class Trainer:

    def __init__(self, m):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture, tracking variables etc.
        """
        if m == 2:
            # Define hyperparameters
            self.epochs = 2
            self.batch_size = 32
            self.learning_rate = 5e-4
            self.early_stop_count = 4

            # Architecture

            # Since we are doing multi-class classification, we use the CrossEntropyLoss
            self.loss_criterion = nn.CrossEntropyLoss()
            # Initialize the mode
            self.model = ResNetModel(num_classes=10)
            # Transfer model to GPU VRAM, if possible.
            self.model = to_cuda(self.model)

            # Define our optimizer. SGD = Stochastich Gradient Descent
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                            self.learning_rate)
        
        else:
            # Define hyperparameters
            self.epochs = 11
            self.batch_size = 64
            self.learning_rate = 5e-3
            self.early_stop_count = 4

            self.weight_decay = 0.0001

            # Architecture

            # Since we are doing multi-class classification, we use the CrossEntropyLoss
            self.loss_criterion = nn.CrossEntropyLoss()
            # Initialize the mode
            self.model = DeepCNNModel(image_channels=3, num_classes=10)
            self.model.apply(init_weights)
            # Transfer model to GPU VRAM, if possible.
            self.model = to_cuda(self.model)


            # Define our optimizer. SGD = Stochastich Gradient Descent
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                lr=self.learning_rate,
                                                eps=1e-7,
                                                weight_decay=self.weight_decay)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)

        self.validation_check = len(self.dataloader_train) // 2

        # Tracking variables
        self.VALIDATION_LOSS = []
        self.TEST_LOSS = []
        self.TRAIN_LOSS = []
        self.TRAIN_ACC = []
        self.VALIDATION_ACC = []
        self.TEST_ACC = []

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()

        # Compute for training set
        train_loss, train_acc = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion
        )
        self.TRAIN_ACC.append(train_acc)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC.append(validation_acc)
        self.VALIDATION_LOSS.append(validation_loss)
        print("Current validation loss:", validation_loss, " Accuracy:", validation_acc)
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC.append(test_acc)
        self.TEST_LOSS.append(test_loss)

        self.model.train()

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss < previous_loss:
                return False
            previous_loss = current_loss
        return True

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        self.validation_epoch()
        for epoch in range(self.epochs):
            # Perform a full pass through all the training samples
            for batch_it, (X_batch, Y_batch) in enumerate(self.dataloader_train):
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = to_cuda(X_batch)
                Y_batch = to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()
                
                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                 # Compute loss/accuracy for all three datasets.
                if batch_it % self.validation_check == 0:
                    self.validation_epoch()
                    # Check early stopping criteria.
                    if self.should_early_stop():
                        print("Early stopping.")
                        return
    
    def visualize (self):
        image = plt.imread("horse.jpg")
        image = to_tensor(image)
        image = normalize(image.data, mean, std)
        image = image.view(1, *image.shape)
        image = nn.functional.interpolate(image, size=(256,256))
        image = to_cuda(image)

        # print(self.model.model)        
        # print(self.model.model.children())

        #Save weights visualization
        weights = self.model.model.conv1.weight.data
        #print(weights.shape)
        torchvision.utils.save_image(weights, "weights_first_layer.png")

        #Save First Layer Activations visualization
        first_layer_out = self.model.model.conv1(image)
        #print(first_layer_out.shape)
        to_visualize = first_layer_out.view(first_layer_out.shape[1], 1, *first_layer_out.shape[2:])
        #print(to_visualize.shape)
        torchvision.utils.save_image(to_visualize, "filters_first_layer.png")

        #Pass image trought all layers but the last 2
        for name, child in self.model.model.named_children():
            if name not in ['avgpool', 'fc']:
                #print("Passing image through layer ", name)
                image = child(image)

        #Save Last Conv. Layer Activations visualization
        to_visualize = image.view(image.shape[1], 1, *image.shape[2:])[:64]
        torchvision.utils.save_image(to_visualize, "filters_last_layer.png")
        

        return


if __name__ == "__main__":
    
    print("Training ResNet18")
    trainer = Trainer(m = 2)
    trainer.train()

    trainer.visualize()

    os.makedirs("plots", exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.VALIDATION_LOSS, label="Validation loss")
    plt.plot(trainer.TRAIN_LOSS, label="Training loss")
    plt.plot(trainer.TEST_LOSS, label="Testing Loss")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss_ResNet18.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(trainer.VALIDATION_ACC, label="Validation Accuracy")
    plt.plot(trainer.TRAIN_ACC, label="Training Accuracy")
    plt.plot(trainer.TEST_ACC, label="Testing Accuracy")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy_ResNet18.png"))
    plt.show()

    print("ResNet18 Results")
    print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])
    print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])
    print("Final training accuracy:", trainer.TRAIN_ACC[-trainer.early_stop_count])

    print("Final test loss:", trainer.TEST_LOSS[-trainer.early_stop_count])
    print("Final validation loss:", trainer.VALIDATION_LOSS[-trainer.early_stop_count])
    print("Final training loss:", trainer.TRAIN_LOSS[-trainer.early_stop_count])

    #Training DeepCNN for comparison
    print("Training Deep CNN")
    trainer_cnn = Trainer(m = 1)
    trainer_cnn.train()

    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.VALIDATION_LOSS, label="[ResNet18] Validation")
    plt.plot(trainer.TRAIN_LOSS, label="[ResNet18] Training")
    plt.plot(trainer.TEST_LOSS, label="[ResNet18] Test")
    plt.plot(trainer_cnn.VALIDATION_LOSS, label="[DeepCNN] Validation")
    plt.plot(trainer_cnn.TRAIN_LOSS, label="[DeepCNN] Training")
    plt.plot(trainer_cnn.TEST_LOSS, label="[DeepCNN] Testing")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss_Comparison.png"))
    plt.show()

    print("Deep CNN Results")
    print("Final test accuracy:", trainer_cnn.TEST_ACC[-trainer_cnn.early_stop_count])
    print("Final validation accuracy:", trainer_cnn.VALIDATION_ACC[-trainer_cnn.early_stop_count])
    print("Final training accuracy:", trainer_cnn.TRAIN_ACC[-trainer_cnn.early_stop_count])

    print("Final test loss:", trainer_cnn.TEST_LOSS[-trainer_cnn.early_stop_count])
    print("Final validation loss:", trainer_cnn.VALIDATION_LOSS[-trainer_cnn.early_stop_count])
    print("Final training loss:", trainer_cnn.TRAIN_LOSS[-trainer_cnn.early_stop_count])