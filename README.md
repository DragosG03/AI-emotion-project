# AI-emotion-project
Emotion detection AI project


Link for the data set:
https://drive.google.com/drive/folders/1n4TzORzMWyX_VPTT3Hd0fqikQ5-_pRZD?usp=sharing


Emotion Recognition From Facial Expressions - A Machine Learning Model
 



Project by:
Dragos Grecianu
Jayna Hanslow
Ingvild Brimsholm

Year 2 BSc (Hons) Computer Science 
- Computer Vision and AI
 


















 Table of Contents


1. Introduction - Introducing our project	3
2. Overall description	4
3. Specific Requirements	6
3.1 Functional requirements	6
3.2 Non-functional requirements	6
4. System Features	7
4.1 Hardware Limitations and Characteristics:	7
4.2 List of Features and Functions:	7
4.3 CNN model	8
5. Use cases	10
5.1 Scenario 1: Childcare	10
5.2 Scenario 2: Healthcare	11
5.3 Scenario 3: Behaviour Improvement	12
5.4 Scenario 4: Customer feedback	13
6. Data requirements	14
7. References	15






















1.	Introduction - Introducing our project

For our project, we have chosen to explore emotion recognition from facial expressions. The objective is to develop an AI program using python coding with CNN model capable of detecting emotions by creating a model that can identify various emotions based on facial expressions in images or video frames. The emotions targeted include Happy, Sad, Anger, Contempt, Disgust, Fear, Neutral, and Surprised.

Motivation:
The main motivation behind this program is to contribute to identifying people's feelings, particularly for businesses, healthcare,  research and education. This could be in the context of autism, where understanding other people's emotions may be challenging, but our intellectual program would offer a solution to help with that. 

Our aim:
We aim to provide a valuable AI machine learning tool that helps in understanding and learning from facial expressions.
This program can serve a crucial role in diverse settings, including research in businesses and healthcare. For example, in a medical context, it can assist in understanding the emotions of non-speaking patients, it would offer a tool for parents seeking to accurately understand their children's needs and behaviour. For psychology purposes, such as therapy sessions or polygraph tests, the program can be very useful for the same reason. Additionally, businesses can use the program to gather feedback from customers, and this will help them to improve their understanding of customer emotions. 
This program can also work like a tool to help people who may struggle recognising emotions, such as “alexithymia”. This machine learning program can help them learn more about themselves, improve the way they express their emotions, and understand other people's feelings.





2.	Overall description

 Our project software will be based in the Python programming language to read facial expressions and figure out emotions like happiness, sadness, anger, and more from pictures or videos.
The emotion recognition program will require a CNN model. CNN stands for Convolutional Neural Network (CNN), which is a supervised machine learning algorithm. As explained in Techtarget, A neural network is “...a system of hardware and software designed to recognize patterns and operate similar to neurons in the human brain.” For example, the CNN has neurons, just like the human brain, to process the information it's given. It is mostly used in image recognition, object classification and pattern recognition. (TechTarget, 2023).  A CNN algorithm model will be set up with the correct settings to classify and categorize images.
 A CNN model requires three layers. The convolutional layer, the pooling layer, and the fully connected layer. The last step is training the CNN model. By “training”, we mean testing and teaching the CNN model what the different emotions are and how to differ between them, by feeding it the dataset we have gathered. For this we need to have epochs. As mentioned on Simplilearn, “machine learning models are trained with specific datasets passed through the algorithm”. It depends on how many epochs are set, but one epoch means the dataset is passed through the model once. To optimize the system to be more precise it helps to run the dataset through more epochs for better predictions accuracy. 
The dataset that the model will use in its analysis consists of images placed in different folders and categorized after their facial expression. This data will be stored in a Google Drive location where the model will have the necessary access to the dataset.
After the CNN model has been created, trained, and verified, the second part of the code takes sample images and runs them through the CNN model to make a prediction on the image dataset. It also gives each image a label with the prediction of its emotion category, as defined by the training dataset.
The third part will take a dataset of images and run them through the CNN model to make predictions on all the facial expressions on the images. The last section will also show the results of how correct the algorithm has assessed that its predictions are.

This program is a beneficial tool for business, healthcare, and education. It helps us understand emotions better, supports research, and provides solutions. It works by analysing facial data in the form of images and classifying the differences between emotions. 
The program will do analysis of both pictures and video material, making it flexible. The software will be split into three different parts. The first module will read facial features input. The second module of the logic will do the deep learning algorithms that will be utilized to analyse and categorise different emotions, primarily the CNN algorithm. The third module of our logic will show the results of the analysis and categorization of the input. 






























3.	Specific Requirements

In this section we talk about our functional and non-functional requirements for our emotion detection of facial recognition machine learning program. 

3.1 Functional requirements
 
●	A CNN algorithm to analyse and detect facial emotions from images. 
●	Dataset of Images of different emotions: This is to give the CNN model the data necessary for analysing and recognising different facial features, such as eyes, nose, eyebrows, mouth, forehead, and cheeks. So as to be able to identify facial expressions and emotions.
●	The CNN model will require the correct settings to organise the different emotions into defined categories. 
●	Users should be able to take a picture or video of their face, feed it to the CNN model program and get a result back on which emotion it shows. 

3.2 Non-functional requirements 

●	Users should be able to get a result of emotion within 3 seconds. 
●	The program should give an accurate prediction of which emotion it is. 
●	The program should give a stable and reliable result with a high prediction assessment score.


4.	System Features
In this section we talk about the system features of our project - The hardware limitations and characteristics (hardware requirements), List of features and functions (CNN model training, prediction and visualisation and its requirements), CNN model (what is it, how it works, four key layers, model construction, training process, and rules set for model optimisation).

4.1 Hardware Limitations and Characteristics:
Hardware Requirements:
●	A machine with a GPU is recommended for faster training. Sufficient RAM to handle the dataset and model. Preferably a newer model, since they have more computing power than older models.

4.2 List of Features and Functions:
CNN Model:
●	Purpose: Learn ordered features from input images for emotion classification. 
●	Behaviour: Consists of convolutional layers, max pooling layers, and fully connected layers.
●	Requirements: Properly formatted input images.

Training:
●	Purpose: Train the CNN model on the provided dataset. 
●	Behaviour: Utilises an image data generator for real-time data augmentation during training.
●	Requirements: Labelled image dataset, computing resources.

Prediction and Visualization:
●	Purpose: Predict the emotion label of a new image and visualise the results. 
●	Behaviour: Uses the trained model to predict emotion labels and displays images with predictions.
●	Requirements: Trained model, new image for prediction.

4.3 CNN model

What is CNN?
CNN, which stands for convolutional neural network, is a machine learning algorithm designed for image processing and classification tasks.

How CNN Works:
At its core, CNNs use different layers to understand images and then learn from them. These layers play a crucial role in pulling out important features and patterns from images, creating a learning path. This model is then used to categorise images with the help of fully connected layers.

Four layers in CNN:
1.	Convolutional Layer:
●	Function: Extracts local patterns using convolutional filters. This means the layer finds patterns in an image, like edges and textures.
●	Description: The first layer in a CNN is responsible for detecting low-level features like edges and textures. It's the first layer in a CNN and is great at understanding the basic features in an image.
2.	Max Pooling Layer:
●	Function: Reduces spatial dimensions, retains important features. 
●	Description: Follows convolutional layers, down-sampling feature maps.
3.	Flatten Layer:
●	Function: Converts 2D arrays to a 1D array. 
●	Description: Prepares the data for input into fully connected layers.
4.	Fully Connected Layer:
●	Function: Makes predictions based on learned features. The last layer makes decisions based on what it has learned.
●	Description: The final layer responsible for classification.

What have we done in our model?
●	Model Architecture: Created a CNN with convolutional layers, max-pooling layers, flattening layer, and fully connected layers for emotion classification. Each layer has a specific job, like understanding patterns, simplifying information, and making decisions. 
●	Training: Trained the model on an emotion dataset using data augmentation.

What rules are set and why?
●	Loss Function: 
-	Categorical Cross Entropy - Suitable for multi-class classification tasks. Categorical Cross Entropy is good for figuring out different emotions. (Towards Data Science, 2020).
-	Optimizer: Adam - Adaptive learning rate optimization algorithm. An optimizer called Adam, is added to the program to help the model learn better. 
●	Metrics: Accuracy - Measures the model's performance during training and evaluation.
5.	Use cases

In this section we talk about some scenarios and examples of where our emotion detection project can be beneficial in childcare, healthcare, and behaviour improvement.

5.1 Scenario 1: Childcare
The User: Parent(s) caring for a non-speaking child
Preconditions: The child  is expressing distress, and the parent(s) is not sure of the cause.
Steps:
1.	The parent(s) uses our emotion recognition system with AI to analyze the child's facial expressions.
2.	The system can identify possible emotions such as the child being hungry, needing the bathroom, health issues, or wanting attention.
3.	The parent(s) can focus on the specific needs of the child, creating a more peaceful environment and making the child happier.
Alternative Paths:
●	If the system is not able to detect a clear emotion, the parent(s) could receive suggestions on different things to do in that situation by checking things such as hunger, sadness, and subtle distress.
●	In case of continued distress, the system may recommend consulting a healthcare professional.
Post Conditions: The child's needs are addressed, leading to a more peaceful household and a satisfied child.



5.2 Scenario 2: Healthcare
User: Medical professionals caring for non-communicative patients
Preconditions: Patients are unable to verbally communicate their emotions or feelings due to their condition.
Steps:
1.	Healthcare professionals use our emotion recognition system to analyse facial expressions of the patients.
2.	Our system then helps to identify the possible causes of pain or any discomfort, anxiety, or other emotions.
3.	Medical workers can then focus on helping and caring for their patients based on the emotional signs showing on the system and with this, improving patients' understanding and needs.
Alternative Paths:
●	If the system detects a potential health concern, it could trigger additional diagnostic procedures.
●	Continuous monitoring can show insights into the effectiveness of the treatment plans.
Post Conditions: Better communication between medical staff and patients, leading to an improved patient care.









5.3 Scenario 3: Behaviour Improvement
User: Individuals seeking personal development
Preconditions: Users want to understand and improve their emotional expressions for better relationships with other people.
Steps:
1.	Users engage with our emotion recognition system to analyse their facial expressions in various situations.
2.	The system delivers feedback on the emotion that is shown to the user. The system will then suggest ways to improve emotional intelligence.
3.	Users can then apply the suggested ways to improve on emotional expression.
Alternative Paths:
●	The system may offer real time feedback in social scenarios for immediate improvement.
●	Personalized training sessions based on identified areas for improvement.
Post Conditions: Users gain better emotional awareness, positively impacting their relationships and personal growth.







5.4 Scenario 4: Customer feedback
User: Businesses seeking to improve their customer service.

Preconditions: Businesses wanting to understand how good their customer service is and to identify if their customers had a pleasant visit.

Steps:

1.	Businesses can set up our emotion detection system in their facilities. 
2.	The system will detect the customers' emotion when they leave the facility.
3.	The businesses will then get a result from the system. Thus, the businesses can take the appropriate action to improve on their overall facility experience and/or customer service. 

Alternative paths:

●	If the customer feedback is already positive the AI can still make extra suggestions such as online advertising based on exactly what the customers liked the most to bring more customers onto the business.

Post conditions: Businesses can improve their customer experience. By receiving a good reputation, businesses can increase their branding with reliability and supportiveness. 
These scenarios show different use cases, possible challenges, and outcomes.





6.	Data requirements 

In this section we summarise the data requirements - (the format, data structures and constraints).

●	Format: Images of human faces with emotions.
●	Data structures: The images are in separate folders based on the emotional facial expression on the image. 
●	Constraints: The images must be put in the correct folder with the correct label/category. If the images are in the wrong category, the model will fail to give an accurate result. Say for example you put happy images in the sad category, or angry images in the surprised category. The model will then interpret the training data wrong because it's in the wrong category, therefore giving wrong results. The model needs rules, therefore it's very important to label the images correctly. For better quality of the result and a more reliable prediction it is helpful to have a bigger dataset and more epochs. 






















7.	References

In this section we have listed out the references that we used to help with our project and writing work for CNN, epoch, and loss function in machine learning. 


1.	TechTarget. (2023). What are Convolutional Neural Networks? [Online]. TechTarget. April. Available at: https://www.techtarget.com/searchenterpriseai/definition/convolutional-neural-network (Accessed: 29 November 2023).

2.	Simplilearn. (2023). What is epoch in machine learning? [Online]. Simplilearn. 7 November. Available at: https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-epoch-in-machine-learning. (Accessed: 29 November 2023).

3.	Towards Data Science. (2020). Cross-Entropy Loss Function. [Online]. Towards Data Science. 2 October 2020. Available at: https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e (Accessed: 1 December 2023).
