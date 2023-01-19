# Siamese Network Face Recognition
Training and using a Siamese network model on a small dataset (e.g., CALFW) for face verification and face recognition.
(Xception was chosen as the backbone of the Siamese network)
![image](https://user-images.githubusercontent.com/109360168/212962643-476fbc15-e864-48ff-b2ae-1990ed818f6f.png)

## Preparing your own dataset
Your dataset should follow the directory structure shown below and all images in your dataset should have the same heights and widths.
```
 path/to/dataset/
                id0/                
                    0.jpg
                    1.jpg
                    ...
                id1/              
                    0.jpg
                    1.jpg
                    2.jpg
                    ...
                ...
```

## Training
input_size: size (height or width) of input image
```
python train.py \
--learning_rate 0.0002 \
--input_size 128 \
--batch_size 64 \
--epochs 40 \
--data_path path/to/training/dataset/ \
--checkpoint_path path/to/save/weights/
```

## Testing
Input with two face images, the model will output the similarity (between -1 and 1) of two faces. You can set a proper threshold for face verification and face recognition.
```
python test.py \
--input_size 128 \
--checkpoint_path path/to/load/weights/ \
--img_path1 path/to/img1/ \
--img_path2 path/to/img2/
```
