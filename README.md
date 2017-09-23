# 5-char-real-number-recognition

[![license](https://img.shields.io/badge/license-MIT%20License-blue.svg)](https://opensource.org/licenses/MIT)

Recognition using segmentation and convnet.

## How to use

1. Run "src/segment_expriment.py" to get familiar with the dataset.

2. "src/expriment2.py" shows how to use the segmentation function. 

3. Run "src/create_datasets.py", "src/create_labels.py" to get croped dataset. 

4. Run "src/train.py" to train the cnn and get pretrain model. ie. "graph" and "metadata" of tensorflow.

5. Run "src/test.py" to achieve 5-character real number recognition.

### **Try to use segment function**

```python
img = segment.Image(img)
list_crop_img, list_crop_loc = img.find_char()
```

## Screen Shot

### Segmented Image

![Segmented Image](https://github.com/JaveyWang/5-char-real-number-recognition/blob/master/image_github_to_markdown/segment_expriment2.png?raw=true)

### Real Number Recognition

![Real Number Recognition](https://github.com/JaveyWang/5-char-real-number-recognition/blob/master/image_github_to_markdown/test_recognition.png?raw=true)
