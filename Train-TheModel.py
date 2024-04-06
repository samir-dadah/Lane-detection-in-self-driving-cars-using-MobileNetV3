codes = np.array(['back', 'left','right'],dtype=str)

carla = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items = my_get_image_files,
                   get_y = label_func,
                   splitter = FuncSplitter(lambda x: str(x).find('validation_set')!=-1),
                   item_tfms=[SegmentationAlbumentationsTransform(albu_transform)])

dls = carla.dataloaders(Path(DATA_DIR), path=Path("."), bs=2)

dls.show_batch(max_n=6)


from fastseg import MobileV3Small

model = MobileV3Small(num_classes=3, use_aspp=True, num_filters=8)


torch.save(learn.model, '/content/drive/MyDrive/le.h5')


import torch
mo = torch.load('/content/drive/MyDrive/lane.h5')
mo.eval()


learn = Learner(dls, model, metrics=[DiceMulti()])

learn.fine_tune(5)

learn.show_results(max_n=6, figsize=(7,8))
