# Fine-tuning-Performance-RNN-to-generate-classical-music

This is my research project for one of the modules I'm taking for my Master degree.  
My goal is to fine-tune the Performance RNN, a model from Magenta project, to generate legit classical music.   
The model was fed with 240 classical music scores in MusicXML file formats. All of them are public domain.  

# Result
I trained the model for 40k checkpoint.   
The dataset was divided into 2 datasets, training and evaluation, with ratio of 90:10. Batch size of 48 was used, with drop rate of 50 percent.  
The blue line is the training set.  
The orange line is the evaluation set.  

<img src="https://user-images.githubusercontent.com/94357278/232262180-f10d816a-c7d3-4641-8e21-44646ed0f853.jpg" alt="metrics_accuracy" width="500" height="300">
The graph above show the accuracy. The accuracy of the training set is 93.8 percent, and 93.2 percent for the evaluation set.   


<img src="https://user-images.githubusercontent.com/94357278/232262134-4da79b2d-1233-4457-b6f4-dd433d81c4ef.jpg" alt="loss" width="500" height="300">
The graph above show the loss value. The loss value of the training set is 0.1895 percent, and 0.2686 percent for the evaluation set.

# Generated Music Samples

https://user-images.githubusercontent.com/94357278/232325485-83642232-7d7d-40e2-a5bf-681c7fd35bdf.mov

Sample 1

https://user-images.githubusercontent.com/94357278/232325495-6156a4c2-85a7-46ce-becd-d6ddf217b530.mov

Sample 2

https://user-images.githubusercontent.com/94357278/232325498-3688a877-2446-49b0-a9b5-6a809296ea6c.mov

Sample 3

# Further Use

I've uploaded the .mag file for the fine-tuned model, so it can be used further.   
https://github.com/sakan811/Fine-tuning-Performance-RNN-to-generate-classical-music/blob/main/classical_fine-tuned_performance_rnn.mag
