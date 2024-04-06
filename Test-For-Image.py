import cv2
img = cv2.imread('/content/drive/MyDrive/01710.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


learn = Learner(dls, mo, metrics=[DiceMulti()])

import numpy as np
plt.imshow(np.array(learn.predict(img)[0]))

def get_pred_for_mobilenet(model, img_array):
    with torch.no_grad():
        image_tensor = img_array.transpose(2,0,1).astype('float32')/255
        x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
        model_output = F.softmax( model.forward(x_tensor), dim=1 ).cpu().numpy()
    return model_output



learn.model.eval();


plt.imshow(get_pred_for_mobilenet(learn.model,img)[0][2])


%timeit get_pred_for_mobilenet(learn.model,img)


import copy
back, left, right = get_pred_for_mobilenet(learn.model,img)[0]
def ld_detection_overlay(image, left_mask, right_mask):
    res = copy.copy(image)
    res[left_mask > 0.8, :] = [255,0,0]
    res[right_mask > 0.8, :] = [0,255,0]
    return res
plt.imshow(ld_detection_overlay(img, left, right))
