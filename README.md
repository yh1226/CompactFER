# CompactFER
The implementation of CVPR 2018 FER paper: A Compact Deep Learning Model for Robust Facial Expression Recognition

I train the model with basic emotion of RAF dataset, and the images were only the face position cropped from origin picture. 
                                        ![image](https://github.com/yh1226/CompactFER/blob/master/img/test_0005_aligned.jpg)    ![image](https://github.com/yh1226/CompactFER/blob/master/img/test_0010_aligned.jpg)     ![image](https://github.com/yh1226/CompactFER/blob/master/img/test_0011_aligned.jpg)
 
I have not exploited the illumination normalization strategy mentioned in the paper and any data agumentation, the best acc achieved to 78% in the RAF's test data.
![image](https://github.com/yh1226/CompactFER/blob/master/img/lossVSepoch2.png)  ![image](https://github.com/yh1226/CompactFER/blob/master/img/accuracyVSepoch2.png)

