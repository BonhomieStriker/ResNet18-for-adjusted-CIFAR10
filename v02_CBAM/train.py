import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from resnet import ResNet18
from dataset import CIFAR10


# choose whether to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters setting
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")
args = parser.parse_args()

# Hyper parameters
EPOCH = 300
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.1

trainset = CIFAR10(train=True) #training set processing
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)   #generate a shuffled batch

testset = CIFAR10(train=False) #test set processing
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
# Labels of CIFAR-10
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'fish', 'flowers')

# Model definition-ResNet
net = ResNet18().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  #Cross-entrophy
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #mini-batch momentum-SGD，L2-norm

# Training
if __name__ == "__main__":
    best_acc = 85  # initialize best test accuracy
    print("Start Training, Resnet-18!")
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                if epoch < 100:
                    optimizer.param_groups[0]['lr'] = 0.1
                elif epoch < 200:
                    optimizer.param_groups[0]['lr'] = 0.01
                else:
                    optimizer.param_groups[0]['lr'] = 0.001

                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # data preparation
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print loss and accuracy
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # test accuracy after one epoch
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # find the most accurate class (index of outputs.data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('Test accuracy of classification：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # write the test accuracy into acc.txt
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # write the best test accuracy record into best_acc.txt
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
