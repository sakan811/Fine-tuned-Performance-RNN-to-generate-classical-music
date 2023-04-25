# Fine-tuning-Performance-RNN-to-generate-classical-music

This is my research project for one of the modules I'm taking for my Master degree.     
I might continue training the model further or with different hyperparameters, even though the module ends.

My goal is to fine-tune the Performance RNN, a model from Magenta project, to generate legit classical music.   

The model was fed with 240 classical music scores in MusicXML file formats. All of them are public domain.    
The 240 MusicXML files were made of 20 classical music scores with each of them transposed to 12 different keys.   

Only the model trained with 48 batch size and 50 percent drop rate is a part of my research project, other version of the trained models aren't.   

# Result of different hyperparameters
I decide to train the model with different hyperparameters to compare the outcomes.   
All of the models was trained until 40k checkpoint.   
The dataset was divided into 2 datasets, training and evaluation, with ratio of 90:10.

# Result (48 batch size)
This version of fine-tuned Performance is a part of my research project.    
Batch size of 48 was used, with drop rate of 50 percent    .    

The blue line is the training set.  
The orange line is the evaluation set.  

<img src="https://user-images.githubusercontent.com/94357278/232262180-f10d816a-c7d3-4641-8e21-44646ed0f853.jpg" alt="metrics_accuracy" width="500" height="300">
The graph above show the accuracy. The accuracy of the training set is 93.8 percent, and 93.2 percent for the evaluation set.   


<img src="https://user-images.githubusercontent.com/94357278/232262134-4da79b2d-1233-4457-b6f4-dd433d81c4ef.jpg" alt="loss" width="500" height="300">
The graph above show the loss value. The loss value of the training set is 0.1895, and 0.2686 for the evaluation set.

# Generated Music Samples
The generated samples were generated using these settings:   
temperature=1    
num_steps=6000   

https://user-images.githubusercontent.com/94357278/232325485-83642232-7d7d-40e2-a5bf-681c7fd35bdf.mov

Sample 1

https://user-images.githubusercontent.com/94357278/232325495-6156a4c2-85a7-46ce-becd-d6ddf217b530.mov

Sample 2

https://user-images.githubusercontent.com/94357278/232325498-3688a877-2446-49b0-a9b5-6a809296ea6c.mov

Sample 3

# Result (32 batch size)
Batch size of 32 was used, with drop rate of 50 percent.      

The blue line is the training set.     
The orange line is the evaluation set.     

<img src="https://user-images.githubusercontent.com/94357278/234107533-47c10fc4-08c0-47d4-a38a-312d95b4a3dd.jpg" alt="loss" width="500" height="300">
The graph above show the accuracy. The accuracy of the training set is 96.3 percent, and 89.1 percent for the evaluation set.   

<img src="https://user-images.githubusercontent.com/94357278/234107514-04ae8c65-ae8e-4d43-89fe-f8cb01a60487.jpg" alt="loss" width="500" height="300">
The graph above show the loss value. The loss value of the training set is 0.1242, and 0.5902 for the evaluation set.

# Generated Music Samples
The generated samples were generated using these settings:   
temperature=1    
num_steps=6000   

https://user-images.githubusercontent.com/94357278/234112926-98e530b5-238a-4403-a769-c49aa7eb34cf.mov   

Sample 1   

https://user-images.githubusercontent.com/94357278/234112936-463f1229-74ce-4cc1-9af8-db6629255d15.mov   

Sample 2   

https://user-images.githubusercontent.com/94357278/234112949-1abe61fd-14db-49a0-8d5f-57673a1ee1ee.mov   

Sample 3   

# Further Use

I've uploaded the .mag file of the fine-tuned model, so it can be used further.   
48 batch size:   
https://github.com/sakan811/Fine-tuning-Performance-RNN-to-generate-classical-music/blob/main/classical_fine-tuned_performance_rnn.mag   

You can look at this page for the tutorial on how to use the Performance RNN   
https://github.com/magenta/magenta/tree/main/magenta/models/performance_rnn
