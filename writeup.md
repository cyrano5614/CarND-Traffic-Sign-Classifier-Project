# **Traffic Sign Recognition** 

---

## **Goal / steps**

* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[image1]: ./pics/graph1.jpg 
[image2]: ./pics/different.jpg
[image3]: ./pics/compare.jpg
[image4]: ./pics/graph2.jpg
[image5]: ./pics/architecture.jpg
[image6]: ./pics/test_images.jpg
[image7]: ./pics/sofrmax_text.jpg
[image8]: ./pics/softmax_image.jpg

---

###Data Set Summary & Exploration

The data set provided initially had 34,799 training sets, 4,410 validation sets, and 12,630 testing sets.  This made the total data sets provided at 51,839, which was made up of 67% training sets, 9% validation sets, and 24% testing sets.  Each image was 32 x 32 with 3 color channels of RGB values each with 0-255 range.  Total number of classes to identify was 43.

![alt text][image1]

From the histogram above, it was apparent that the training dataset that was provided showed significant imbalance between the classes.  For the model to perform well on accuracy and generalize well, the dataset needs to be balanced for the model to be not biased as much as possible.

![alt text][image2]

As the images show, they all differ in brightness, contrast, and intensity which will also need to be adjusted to make accurate model.


###Preprocess the data

At first, the data was ran straight into the LeNet model with around 80% validation accuracy and the need for data preprocessing was apparent.  Through research, I realized using single channel of color and simplifying the model generally out weighs the benefit of keeping all 3 channels of colors.  It also made the following image manipulations much simpler and faster.

I converted the images to grayscale using built in function in opencv library which uses 0.299 * R + 0.587 * G + 0.114 * B to make three RGB channels to single channel.

![alt text][image3]

After observing numerous method to manipulate and change image for better modeling, I have decided to use localised histogram built in skimage library.  I chose to use the function over other various functions I have tried because it had build-in normalization and the ability to enhance the regions in image that are darker or lighter than most of the image.

Preprocessing the data still was not achieving the accuracy largely due to the imbalance of the dataset I have previously mentioned.  To overcome this, I decided to generate additional augmented data to neutralize the imbalance of the dataset.

![alt text][image4]

As seen in the example above, I created the augmented data by selecting the classes that lacks the numbers compared to the other classes.  From there, I generated same image and class but with a slight rotation to simulate different situations.  The angles of the rotation was chosen randomly between [-10, -7, -5, -2, 2, 5, 7, 10].

###Design and Test a Model Architecture



My final model consisted of the following layers:

![alt text][image5]

The figure above represents the model architecture I have decided to use.  This LeNet model have been widely used for CNN especially with MNIST data.  It consists of 2 convolutional layers with 3 fully connected layers.  Each convolutional layer uses 5x5 filter with depth of 6 and 16 respectively.  Both convolutional layer also goes through "RELU" activation function and max-pooling to make input representation smaller, help prevent overfitting, and locate object anywhere in the image.  Each fully connected layers go through RELU also except the last layer as it proceeds to classify the labels.  Between the fully connected layers and the last convolutional layer, I chose to use dropout function to minimize overfitting.


To train the model, I used an Adam optimizer which is more convinient and efficient than gradient descent in tensorflow as it has built-in learning rate decaying and other useful functions.  I used batch size of 150 with 30 epoch as I frequently saw the accuracy stop near 30.  After numerous tuning,  I have settled to use learning rate of 0.003.  To furthur help minimize overfitting, I decided to integrate L2 regularization with beta value of 0.001.


My final model results were:
* validation set accuracy: 95.9% 
* test set accuracy: 93.6%

I chose to use LeNet architecture used previously in coure work which I was familiar with.  I chose to use the architecture because it has proven track record with MNIST dataset and other image classfication applications.  I made numerous adjustments tuning hyperparameters and dropout rates to minimize the overfitting and perform well on both valid and testing sets.  Additional dropout layer was added on 2nd convolutional layer to further prevent overfitting.  

 

###Test a Model on New Images

I chose 5 different German traffic signs to test my model and predict the correct label.  I chose few images that I considered easy which was the speed limit signs, and the few that might be more difficult which was double curve and road work sign.  Given that the sign shapes were more complicated and narrow clustered, I thought this might cause model to have more difficulty.

![alt text][image6]

Here are the results of the prediction:


![alt text][image7]


Thoe model only predicted 3 out of 5 signs correctly which left us with 60% accuracy.  Compared to the test accuracy of 93%, it was comparably lower.  


The first image was 'Stop' Sign which was predicted correctly with relatively higher probability than other choices such as 'No entry' sign or 'Road work' sign, which came 2nd and 3rd respectively.  

The second image was speed limit 50km sign which was predicted correctly with other speed limit signs coming in on following probabilities.

The third image was quiet a surprise for me as the model failed to predict speed limit 60km and instead guessed it to be speed limit 120km.  The correct prediction was pushed to 3rd rank. 

The fourth image was correctly predicted as 'Road work' which was a surprise as my original guess was that the model would have difficulty trying to predict this sign.  Dangerous curve to the right sign and bumpy road sign came 2nd and 3rd respectively.

The fifth and last image was double curve sign which the model predicted wrong.  The model predicted right-of-way at the next intersection which was understandable given the shape, but not seeing double curve on any of the top 5 softmax probabilies was odd.

![alt text][image8]





