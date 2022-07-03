import cv2
import lib as library
import network
import torch

# define a video capture object

vid = cv2.VideoCapture(1)
model = network.MyNetwork()
network_state_dict = torch.load('modelMNIST.pth')
model.load_state_dict(network_state_dict)

while (True):
    ret, frame = vid.read()

    cv2.imshow('frame', frame)

    cv2.imwrite('live/digit/Frame.jpg', frame)
    digit = library.loadCustomData('live', 1)
    dataN = enumerate(digit)
    batch_idx, (example_data, example_targets) = next(dataN)
    output = model(example_data)

    for i in range(len(output)):
        print("Image {0} and values are {1}".format(i + 1, output[i]))
        print(library.retMaxIndex(output[i]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()