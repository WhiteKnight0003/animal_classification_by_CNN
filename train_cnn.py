from dataset import AnimalDataset
from model import SimpleCNN
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor, Compose, RandomAffine, ColorJitter
import torch
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

# >>> THÊM 4 DÒNG NÀY LÊN ĐẦU FILE <<<
import os
os.environ["MPLBACKEND"] = "Agg"          # tránh WX/GTK
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # phòng khi có Qt
os.environ["PYTHONWARNINGS"] = "ignore"      # (tuỳ chọn) làm gọn log

import matplotlib.pyplot as plt
import os
import numpy as np
import shutil
import cv2
from torchsummary import summary 
from tqdm.autonotebook import tqdm 
from torch.utils.tensorboard import SummaryWriter 

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    

    figure = plt.figure(figsize=(20, 20))
    
    plt.imshow(cm, interpolation='nearest', cmap="ocean") 
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--epochs","-e", type = int, default=10, help ="Number of epochs" )
    parser.add_argument("--batch_size", "-b",type = int, default=4, help ="batch size" ) 
    parser.add_argument("--image_size","-i", type = int , default= 224, help = "image size")
    parser.add_argument("--root","-r", type = str , default= './dataset', help = "Root")
    parser.add_argument("--logging","-l", type = str , default= './tensorboard_file')  
    parser.add_argument('--trained_models', '-tr', type=str, default='./trained_models')
    
    parser.add_argument('--checkpoint', '-c', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ =='__main__':
    args = get_args() #  

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print("cpu")
    print(args.batch_size)
    print(args.epochs)

    train_transforms = Compose([ 
        RandomAffine(
            degrees=(-10,10), # xoay 
            translate=(0.1,0.1), # dịch
            scale=(0.9,1.1), # zoom in , zoom out
            shear=10 # xiên
        ),

        # thay đổi màu sắc 
        ColorJitter(
            brightness=0.15,
            contrast=0.4, # độ tương phản 
            saturation= 0.3 , # độ bão hòa
            hue=0.05, # độ nhòe
        ),

        Resize((args.image_size, args.image_size)), 
        ToTensor(), 
    ])

    test_transforms = Compose([ 
        Resize((args.image_size, args.image_size)), 
        ToTensor(), 
    ])


    train_dataset = AnimalDataset(root = os.path.join(args.root, 'train'), train = True,transforms=train_transforms)

    # test thử ảnh
    # ima, _ = train_dataset.__getitem__(1000)
    # print(ima.shape) 
    # ima = (torch.permute(ima, (1,2,0))*255).numpy().astype(np.uint8) 
    # ima = cv2.cvtColor(ima, cv2.COLOR_RGB2BGR)
    # print(ima.shape)
    # cv2.imshow('test image', ima)
    # cv2.waitKey(0)
    # exit(0)

    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size=args.batch_size,
        shuffle = True,
        num_workers=2,
        drop_last=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataset = AnimalDataset(root=os.path.join(args.root, 'test'), train = False,transforms=test_transforms)
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size=args.batch_size,
        shuffle = False,
        num_workers=2,
        drop_last=True,
        pin_memory=True if torch.cuda.is_available() else False
    )

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging) # xóa hết cả thư mục
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models) # chưa tồn tại thì tạo

    writer = SummaryWriter(args.logging)


    model = SimpleCNN(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum =0.9)

    # nếu model chưa tồn tại thì train từ đầu , nếu đang train dở thì train tiếp
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_acc"]
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else: 
        start_epoch=0
        best_accuracy =0

    num_iter = len(train_dataloader)
    
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour='green') # chọn màu cho thanh tiến trình
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss_value = criterion(outputs, labels)

            progress_bar.set_description("Epoch {}/{}.  Iteration {}/{} . Loss {:.3f}".format(epoch+1, args.epochs, iter+1, num_iter, loss_value)) 

            writer.add_scalar('Train/Loss', loss_value, epoch*num_iter+iter) 

            # Backward pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []

        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                prediction = model(images)
                indices = torch.argmax(prediction.cpu(), dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(prediction, labels)

        all_labels = [label.item() for label in all_labels]
        all_predictions = [pred.item() for pred in all_predictions]
        # Print classification report

        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), class_names=test_dataset.class_to_idx.keys(), epoch=epoch)
        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {}   Accurcy:  {}".format(epoch+1,accuracy))
        
        writer.add_scalar('Val/Accuracy', accuracy,epoch) 

        checkpoint ={
            "epoch": epoch+1, 
            "model": model.state_dict(),
            "best_acc": best_accuracy,
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))

        if accuracy > best_accuracy:
            checkpoint ={
            "epoch": epoch+1, 
            "model": model.state_dict(),
            "best_acc": best_accuracy,
            "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
            best_accuracy = accuracy
