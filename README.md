# Modify and Fine-tune Performance RNN to Generate Classical Music

This is my research project for one of the modules I'm taking for my Master's degree.     
I might continue with this further, even though my research project finishes.

My goal is to create another version of Performance RNN that can generate classical music that sounds close to human-composed classical music.

The project involved 3 iteration processes that involved model modifications and fine-tuning processes.

## Training
All of the models used are pre-trained models.     

The models were fed with 1329 MIDI and MusicXML files in total.     
40 MusicXML files were from Musescore and they are all public domain.   
The rest was from Aligned Scores and Performances (ASAP) dataset: https://github.com/fosfrancesco/asap-dataset    
     
The dataset will be divided into 2 datasets, training, and evaluation, with a ratio of 90:10.    

## Table of Contents
- [Result of the 1st Iteration (Original Model)](#result-original-model)
- [Result of the 2nd Iteration (1st Modified Model)](#result-modified-model)
- [Further Use](#further-use)

## Result of the 1st Iteration (Original Model)
This is the original pre-trained Performance RNN model that was fine-tuned with the classical music datasets.     
The model was trained for around 191,000 steps.        
 
**Generated Music Samples:**         
- Sample 1: https://drive.google.com/file/d/1QzRQGdM1Xrz06SBvFDHjB2tvfqCOmLkQ/view?usp=sharing          
- Sample 2: https://drive.google.com/file/d/1sh3sMlHLHRtKav0iVs6fUEOBeTwspeH_/view?usp=sharing      
- Sample 3: https://drive.google.com/file/d/1xUncQSNSoyaA13ny0EP-qPmOKZGzYr9z/view?usp=sharing    
    - All of these generated with the parameter num_steps set at 6000.
  
**Training details:**        
**Batch size of 48** was used, with **drop rate** of **50 percent**.    
The rest of the hyperparameters' setting was the default. 

**Changelog:**   
- At around 65,000 steps, the learning rate was changed from 0.001 to 0.002 to make the model learns faster.  
- At around 71,000 steps, the learning rate was changed back to 0.001 as the model's performance worsened.   
  
## Result of the 2nd Iteration (1st Modified Model)
This is the modified pre-trained Performance RNN model that was fine-tuned with the classical music datasets.     
The model was trained for around 21,500 steps. 

**The model was modified by**:     
- Adding the custom loss function to make the model adheres more to the rhythm and harmonic structure of classical music.       
- Adding the early stop function to make the model stops the training when the loss and accuracy from evaluation set don't improve for a certain number of steps.   
- Adding L1 regularization to prevent overfitting even more.
- Adding Cyclical Learning Rate to help mamanging learning rate.

**Generated Music Samples:**         
- Sample 1: https://drive.google.com/file/d/1fjUI6lVJegdN8QacCVUb14jk-cmBhvpY/view?usp=sharing       
- Sample 2: https://drive.google.com/file/d/1NJzaEjIysItNRboGO1cEF1cewlCYxsYE/view?usp=sharing   
- Sample 3: https://drive.google.com/file/d/1bVGBCh7x2QYdFluxvD5_9wO6VeCGd71f/view?usp=sharing 
    - All of these generated with the parameter num_steps set at 6000.

**Abbreviation**
- LR (Learning Rate)
- CLR (Cyclical Learning Rate)
- L1 (L1 Regularization)
- RL (Rhythm Loss)
- HPL (Harmonic Progression Loss)

**Training details:**  
- **Batch size of 48** was used, with **drop rate** of **50 percent**. 
- This model involved CLR.      
    - At first, I set the maximum LR to be 0.001 and the minimum LR to be 0.0001, following the rule from research by Leslie N. Smith.     
    - Later I changed the minimum LR was 0.00025, which was the 1/4 of the maximum LR, following another rule from research, as the above method didn't work well for the model.
    - So the lower-bound was 0.00025, and the upper-bound was 0.001. 
    - The number of steps for each half-cycle in CLR was 470, which is 2 times of the number of iterations for each epoch.   
- L1 scale was set to 0.0001.            
- RL and HPL were used.
    - These were additional loss fuction built on top of the existing one.  
    - The weight was set at 0.5 for both.  
- Early-stop's patience was set at 4,700, which was equal to 20 epochs.                   
    - Originally, early-stop's patience was set at 2,350, which was equal to 10 epochs.    
- The rest of the hyperparameters' setting was the default.   

Before getting to this model, there were many trials and errors involved to find the best value for early-stop, L1, LR, drop-out rate, and RL and HPL's weights for the model.     

The lower-bound and upper-bound for CLR were set based on the rules recommended by the "Cyclical Learning Rates for Training Neural Networks" research by Leslie N. Smith. There are 2 rules from the research, but for this model, I used the rule that said we could find the best learning rate range for CLR by finding the maximum LR that made the model converge and set the minimum LR to be 1/3 or 1/4 of the maximum LR.

The research also suggested about finding the right step size for CLR, which was the 2 - 10 times of the number of iterations of an epoch.

The maximum LR were found during my trial-and-error experimentation to find the best LR.

**Changelog:** 
- If the early-stop activated, but the model showed signs of improvement, I continued the training with another training loop, which started with another CLR cycles. 
- At around 4,000 steps, the early-stop activated, but I saw the potential that the model could improve, so I continued the training and doubled the patience value.
- At around 21,500 steps, the early-stop activated. 


## List of Modified files
- **events_rnn_train.py**
    - **Implemented early stopping**
        - Early stopping takes the loss from evaluation set to determine when the training should stop.
        - You can set the "patience" value.
            - "patience" is the number of steps that the early stopping algorithm will tolerate when it sees that loss from the evaluation set aren't improving anymore.
                - So if you want the algorithm to tolerate for 1,000 steps when the loss of the evaluation set aren't improved before stopping the training, you can set the "patience" to 1,000.           
            - the "patience" need to be set if you want to use early stopping. 
            - To enable Early stopping, this following command line is an example:     
                ```
                --early_stop='patience=1000' \
                ``` 
- **events_rnn_graph.py**
    - **Modified the loss function**
        - Added **Rhythm loss** and **Harmonic progression loss** on top of the original loss function.    
            - To enable Rhythm loss, use this following command line:     
                ```
                --rhythm_loss \
                ``` 
            - To enable Harmonic progression loss, use this following command line:     
                ```
                --harmony_loss \
                ``` 
    - **Added L1 Regularization**
        - Added **L1 Regularization** on top of the original loss function.  
            - To enable L1 Regularization, set the scale value to more than 0. If set to 0, the L1 Regularization won't be used.
            - To enable L1 Regularization, this following command line is an example:     
                ```
                --l1_regular='0.001' \
                ```   
    - **Added Cyclical Learning Rate**  
        - To enable **Cyclical Learning Rate**, set the value for lower-bound, upper-bound, and step size.   
        - This following command line is an example:        
                ```  
                --clr_hparams='lower_bound=0.00025,upper_bound=0.001,step_size=470' \   
                ```     
- **performance_rnn_train.py**
    - Added command line flags for the added functions, so they can be controlled manually.

## Further Use
You can look at this page for the tutorial on how to use the Performance RNN   
https://github.com/magenta/magenta/tree/main/magenta/models/performance_rnn
