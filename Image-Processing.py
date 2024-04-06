
def get_image_array_from_fn(fn):
    image = cv2.imread(fn)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

DATA_DIR = "/content"


x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_label')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_label')

 
def label_func(fn): 
    return str(fn).replace(".png", "_label.png").replace("train", "train_label").replace("val/", "val_label/")


 
sample_fn = x_train_dir + "/" + os.listdir(x_train_dir)[0]
print(sample_fn)
plt.imshow(get_image_array_from_fn(sample_fn));


 
label_fn = label_func(sample_fn)
print(label_fn)
 
plt.imshow(100*get_image_array_from_fn(label_fn)); 



my_get_image_files = partial(get_image_files, folders=["train", "val"])
