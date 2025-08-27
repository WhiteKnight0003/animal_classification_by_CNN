from argparse import ArgumentParser
import torch
from model import SimpleCNN
import cv2
import numpy as np
import torch.nn as nn
from torchsummary import summary # dùng để in ra cấu trúc model của bạn 
 
def get_args():
    parser = ArgumentParser(description="CNN inference")
    parser.add_argument("--image_size", '-i', type = int, default=224, help = 'Image size ')
    parser.add_argument('--image_path', '-p', type = str , default = None, help = "Image path")
    parser.add_argument("--checkpoint", '-c', type= str, default='./trained_models/best_cnn.pt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    category = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

    args = get_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda')
    else:
        device = torch.device('cpu')
        print('cpu')

    model = SimpleCNN(num_classes=10).to(device)
    
    # summary(model, (3,224, 224)) # truyền vào model và kích thước ảnh

    # load check point len
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
    else: # neeus k co checkpoint nao
        print('No checkpoint found!')
        exit(0)

    # test
    model.eval()

    # use open cv and numpy as processing image
    ori_image = cv2.imread(args.image_path) # BGR -> RGB - # giữ ảnh gốc để cuối cùng còn show
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB) # ở phần getitem làm như nào thì đây xử lý ảnh như v - đầu tiên đổi chiều kênh màu
    image = cv2.resize(image,(args.image_size, args.image_size)) # tiếp theo resize - về 224x224
    image = np.transpose(image, (2,0,1))/255.0 # change [h,w,c] -> [c,h,w] and convert pixel to [0, 1]
    image = image[None, :, :, :] # add batch_size - ex [1, 3, 224, 224]
    image = torch.from_numpy(image).to(device).float()


    # softmax function
    softmax = nn.Softmax() # 

    # k tinh gradient ở phần test và val
    with torch.no_grad():
        output = model(image)
        # softmax
        probs = softmax(output) # # array 2d

        # lấy kq ở output hay softmã đều đc vì cái lớn nhất đều là cùng 1 cái

    max_idx = torch.argmax(probs) # find idx of element max value
    predicted_class = category[max_idx]
    print("The test image is about {}  with confident score of {}".format(predicted_class, probs[0,max_idx])) # tensor 2 chiều kiểu [1,10] nên dổi chiều đầu tiên về 0

    cv2.imshow("{} : {:.2f}%".format(predicted_class, probs[0,max_idx]*100), ori_image) 
    cv2.waitKey(0)
