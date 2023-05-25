# Modify and Fine-tune Performance RNN to Generate Classical Music

This is my research project for one of the modules I'm taking for my Master degree.     
I might continue with this further, even though my research project finishes.

My goal is to fine-tune and modify the Performance RNN, a model from Magenta project, to enhance its performance to generate better classical music.

I will compare the original fine-tuned pre-trained Performance RNN model to the modifed and fine-tuned version of the model.    

## Training
All of the model used are pre-trained models.     

The models will be fed with 240 classical music scores in MusicXML file formats. All of them are public domain.      
The 240 MusicXML files were made of 20 classical music scores with each of them transposed to 12 different keys.      

The dataset will be divided into 2 datasets, training and evaluation, with ratio of 90:10.    

Due to my limit computational resource, the batch size need to be reduced to 48 to avoid Out Of Memory.   

## Table of Contents
- [Result (Original Model)](#result-original-model)
- [Generated Music Samples (Original Model)](#generated-music-samples-original-model)
- [Result (Modified Model)](#result-modified-model)
- [Generated Music Samples (Modified Model)](#generated-music-samples-modified-model)
- [Further Use](#further-use)

## Result (Original Model)
This is the original model that was fine-tuned with the classical music datasets.    

**Batch size of 48** was used, with **drop rate** of **50 percent**.        
The rest of the hyperparameters' setting was default.     

The model was trained for 40k checkpoint.                  

The **blue line** is the **training set**.   
The **orange line** is the **evaluation set**.   
 
<img src="https://user-images.githubusercontent.com/94357278/232262180-f10d816a-c7d3-4641-8e21-44646ed0f853.jpg" alt="metrics_accuracy" width="500" height="300">

The graph above show the accuracy. The **accuracy** of the **training set** is **93.8 percent**, and **93.2 percent** for the **evaluation set**.   


<img src="https://user-images.githubusercontent.com/94357278/232262134-4da79b2d-1233-4457-b6f4-dd433d81c4ef.jpg" alt="loss" width="500" height="300">

The graph above show the loss value. The **loss value** of the **training set** is **0.1895**, and **0.2686** for the **evaluation set**.

## Generated Music Samples (Original Model)
The generated samples were generated using these settings:   
temperature=1    
num_steps=6000   

https://user-images.githubusercontent.com/94357278/232325485-83642232-7d7d-40e2-a5bf-681c7fd35bdf.mov

Sample 1

https://user-images.githubusercontent.com/94357278/232325495-6156a4c2-85a7-46ce-becd-d6ddf217b530.mov

Sample 2

https://user-images.githubusercontent.com/94357278/232325498-3688a877-2446-49b0-a9b5-6a809296ea6c.mov

Sample 3

## Result (Modified Model)
**The model was modified by**:     
-Adding the custom loss function that adhere more to the rhythm and harmonic structure of classical music.       
-Adding the early stop function to make the model stops the training when the loss doesn't improve for a certain number of steps or when the loss value reaches the target loss.     

**Batch size of 48** was used, with **drop rate** of **50 percent**.               
The rest of the hyperparameters' setting was default.     
    
**This time:**       
The **orange line** is the **training set**.     
The **blue line** is the **evaluation set**.       
   
Training in progress

## Training Changelog

L1 regularization was used when the model reach 11.5k steps.   
 - To avoid data overfitting. As during this steps, the accuracy of evaluation set was quite lower than the training set.
   
L1 regularization scale was changed from 0.001 to 0.0001 after 18k steps.   
 - The accuracy of the model wasn't improved. The L1 regularization scale was changed to make the accuracy improve.        

L1 regularization scale was changed from 0.0001 to 0.00001 after 28k steps.      
 - The accuracy of the model wasn't improved. The L1 regularization scale was changed to make the accuracy improve.    

L1 regularization scale was changed from 0.00001 to 0.000055 after 35k steps.      
 - The accuracy of the evaluation set was lower that the training set.

L1 regularization scale was changed to 0.00055 after 36k steps.          
 - The accuracy of the evaluation set was lower that the training set.    
 - The accuracy of the model also wasn't improved.

L1 regularization scale was changed back to 0.001 after around 36.3k steps.          
 - To try fixing the evaluation set's accuracy problem.

<!-- <img src="https://user-images.githubusercontent.com/94357278/235756373-ce9cc17d-cf09-438e-a118-b3df759a7dc6.jpg" alt="loss" width="500" height="300">      

The graph above show the accuracy. The **accuracy** of the **training set** is **98.3 percent**, and **96.9 percent** for **the evaluation set**.   

<img src="https://user-images.githubusercontent.com/94357278/235756537-e7c26aab-080f-44c2-8c5e-b762a5956841.jpg" alt="loss" width="500" height="300">

The graph above show the loss value. The **loss value** of the **training set** is **0.0526**, and **0.1596** for **the evaluation set**. -->

<!-- ## Generated Music Samples (32 batch size)
The generated samples were generated using these settings:   
temperature=0.25    
num_steps=6000   

https://user-images.githubusercontent.com/94357278/235761317-36e20018-cc4d-41f9-8072-d30742b951f4.mov

Sample 1   

https://user-images.githubusercontent.com/94357278/235761326-9d6a3753-4d07-475d-a397-05932218edb4.mov

Sample 2   

https://user-images.githubusercontent.com/94357278/235761340-34a82e1d-d50a-4479-9b27-c77d3da2dffd.mov

Sample 3    -->

## List of Modified files
- **events_rnn_train.py**
    - **Implement early stopping**
        - You can set the "mode", "patience", and the "target loss" or "target accuracy" according to the mode
        - There are 2 modes: "min" and "max"
            - "min" mode is for monitoring loss value and target loss has to be set. 
                - training will stop when the target loss is reached or loss isn't improved for a certain number of steps
                - If you want to use "min" mode, the following command line is an example:
                ```bash
                --early_stop='mode=min,target_loss=0.12,patience=10' \
                ```
            - "max" mode is for monitoring accuracy and target accuracy has to be set. 
                - training will stop when the target accuracy is reached or accuracy isn't improved for a certain number of steps
                - If you want to use "max" mode, the following command line is an example:
                ```bash
                --early_stop='mode=max,target_acc=0.9,patience=10' \
                ```
            - the number of steps that the early stopping algorithm will tolerate is defined by the "patience" variable multipied by the "summary_frequency" variable
            - the "summary_frequency" is 10. 
            - the default "patience" is 100. 
                - So if you want the algorithm to tolerate for 1000 steps when the loss value isn't improved before stopping the training, you can set the "patience" to 100.
- **events_rnn_graph.py**
    - **Modify the loss function**
     - Added **Rhythm loss** and **Harmonic progression loss** on top of the original loss function.


## Further Use

I've uploaded the .mag file of the fine-tuned model, so it can be used further.   
Fine-tuned Original Model:   
https://github.com/sakan811/Fine-tuning-Performance-RNN-to-generate-classical-music/blob/main/classical_fine-tuned_performance_rnn.mag       
<!-- 32 batch size:    
https://github.com/sakan811/Fine-tuning-Performance-RNN-to-generate-classical-music/blob/main/32_classical_fine-tuned_performance_rnn.mag    -->

You can look at this page for the tutorial on how to use the Performance RNN   
https://github.com/magenta/magenta/tree/main/magenta/models/performance_rnn
