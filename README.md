# Modify and Fine-tune Performance RNN to Generate Classical Music

This is my MSc dissertation project.       

This project was to improve Performance RNN to generate better classical music with a "human-like" quality.   

A "human-like" quality is the qualities that we see in Human-Composed music, such as "a clear tonality", "a clear chord progression", and "a good sense of rhythm".    

Performance RNN was already trained with classical music, but I thought that the generated classical music still lacked that "human-like" quality.   

I aimed to fine-tune the model further with more classical music to enhance its performance.   

The project involved 3 iterative processes that involved model script modifications, fine-tuning, and evaluation processes.  

## Training
All of the models used are pre-trained Performance RNN models.     

The models were trained with 1,342 MIDI and MusicXML files in total.     
40 MusicXML files were from Musescore.com and they are all public domain.   
The rest was from the Aligned Scores and Performances (ASAP) dataset: https://github.com/fosfrancesco/asap-dataset    
     
The dataset will be divided into 2 datasets, training, and evaluation, with a ratio of 90:10.    

## Results
All of the Performance RNN models from this project could not generate classical music with a "human-like" quality as I expected.   

The only things that were successful were 2 functions that I added to the scripts: Early-stop and Cyclical Learning Rate.

Early-stop helps with preventing overfitting, as the original scripts don't provide this function.   

Cyclical Learning Rate helps with saving training times, as stated in "Cyclical Learning Rates for Training Neural Networks" research by Leslie N. Smith.

## Table of Contents
- [Result of the 1st Iteration](#result-of-the-1st-iteration)
- [Result of the 2nd Iteration](#result-of-the-2nd-iteration)
- [Result of the 3rd Iteration](#result-of-the-3rd-iteration)
- [List of Modified Files](#list-of-modified-files)
- [Tutorial](#tutorial)

## Result of the 1st Iteration 
This is the original pre-trained Performance RNN model that was fine-tuned more classical music dataset.     
The model was trained for around 191,000 steps.        
 
**Generated Music Samples:**         
- [Sample 1](https://github.com/sakan811/Modify-and-fine-tune-Performance-RNN-to-Generate-Classical-Music/blob/main/Generated%20Samples%20(Unquantized)/1st%20model/sample%201.mp3)      
- [Sample 2](https://github.com/sakan811/Modify-and-fine-tune-Performance-RNN-to-Generate-Classical-Music/blob/main/Generated%20Samples%20(Unquantized)/1st%20model/sample%202.mp3)     
- [Sample 3](https://github.com/sakan811/Modify-and-fine-tune-Performance-RNN-to-Generate-Classical-Music/blob/main/Generated%20Samples%20(Unquantized)/1st%20model/sample%203.mp3)  
    - All of these were generated with the parameter num_steps set at 6000.
  
**Training details:**        
**Batch size of 48** was used, with **drop rate** of **50 percent**.    
The rest of the hyperparameters' setting was the default. 

**Changelog:**   
- At around 65,000 steps, the learning rate was changed from 0.001 to 0.002 to make the model learn faster.  
- At around 71,000 steps, the learning rate was changed back to 0.001 as the model's performance worsened.   
  
## Result of the 2nd Iteration
This is the modified pre-trained Performance RNN model that was fine-tuned with a more classical music dataset.     
The model was trained for around 21,500 steps. 

**The model was modified by**:     
- Adding the custom loss function to make the model adhere more to the rhythm and harmonic structure of classical music.       
- Adding the early stop function to make the model stop the training when the loss and accuracy from the evaluation set don't improve for a certain number of steps.   
- Adding L1 regularization to prevent overfitting even more.
- Adding Cyclical Learning Rate to help manage learning rate.

**Generated Music Samples:**         
- [Sample 1](https://github.com/sakan811/Modify-and-fine-tune-Performance-RNN-to-Generate-Classical-Music/blob/main/Generated%20Samples%20(Unquantized)/2nd%20model/sample%201.mp3)       
- [Sample 2](https://github.com/sakan811/Modify-and-fine-tune-Performance-RNN-to-Generate-Classical-Music/blob/main/Generated%20Samples%20(Unquantized)/2nd%20model/sample%202.mp3)   
- [Sample 3](https://github.com/sakan811/Modify-and-fine-tune-Performance-RNN-to-Generate-Classical-Music/blob/main/Generated%20Samples%20(Unquantized)/2nd%20model/sample%203.mp3) 
    - All of these were generated with the parameter num_steps set at 6000.

**Abbreviation**
- LR (Learning Rate)
- CLR (Cyclical Learning Rate)
- L1 (L1 Regularization)
- RL (Rhythm Loss)
- HPL (Harmonic Progression Loss)

**Training details:**  
- **Batch size of 48** was used, with **drop rate** of **50 percent**. 
- This model involved CLR.      
    - At first, I set the maximum LR to be 0.001 and the minimum LR to 0.0001, adapting the rule about finding the best LR range from research by Leslie N. Smith.     
    - Later I changed the minimum LR was 0.00025, which was 1/4 of the maximum LR, adapting another rule from research, as the above method didn't work well for the model.
    - So the lower bound was 0.00025, and the upper bound was 0.001. 
    - The number of steps for each half-cycle in CLR was 470, which is 2 times the number of iterations for each epoch.   
- L1 scale was set to 0.0001.            
- RL and HPL were used.
    - These were additional loss functions built on top of the existing ones.  
    - The weight was set at 0.5 for both.  
- Early-stop's patience was set at 4,700, which was equal to 20 epochs.                   
    - Originally, early-stop's patience was set at 2,350, which was equal to 10 epochs.    
- The rest of the hyperparameters' setting was the default.   

Before getting to this model, there were many trials and errors involved in finding the best value for early-stop, L1, LR, drop-out rate, and RL and HPL weights for the model.     

The lower-bound and upper-bound for CLR were set based on the rules recommended by the "Cyclical Learning Rates for Training Neural Networks" research by Leslie N. Smith. There are 2 rules from the research, but for this model, I used the rule that said we could find the best learning rate range for CLR by finding the maximum LR that made the model converge and set the minimum LR to be 1/3 or 1/4 of the maximum LR.

The research also suggested finding the right step size for CLR, which was 2 - 10 times the number of iterations of an epoch.

The maximum LR was found during my trial-and-error experimentation to find the best LR.

**Changelog:** 
- If the early stop activated, but the model showed signs of improvement, I continued the training with another training loop, which started with another CLR cycle. 
- At around 4,000 steps, the early-stop activated, but I saw the potential that the model could improve, so I continued the training and doubled the patience value.
- At around 21,500 steps, the early-stop activated. 


## Result of the 3rd Iteration 
This is the modified pre-trained Performance RNN model that was fine-tuned with a more classical music dataset.     
The final model was trained for around 1,958 steps. 

**Generated Music Samples:**         
- [Sample 1](https://github.com/sakan811/Modify-and-fine-tune-Performance-RNN-to-Generate-Classical-Music/blob/main/Generated%20Samples%20(Unquantized)/3rd%20model/sample%201.mp3)       
- [Sample 2](https://github.com/sakan811/Modify-and-fine-tune-Performance-RNN-to-Generate-Classical-Music/blob/main/Generated%20Samples%20(Unquantized)/3rd%20model/sample%202.mp3)
- [Sample 3](https://github.com/sakan811/Modify-and-fine-tune-Performance-RNN-to-Generate-Classical-Music/blob/main/Generated%20Samples%20(Unquantized)/3rd%20model/sample%203.mp3)
    - All of these were generated with the parameter num_steps set at 6000, the rest was the default.
    - Sample 1 had the pitch_class_histogram set to the key of C Major.
    - Sample 2 had the pitch_class_histogram set to the key of D Major.
    - Sample 3 had the pitch_class_histogram set to the key of D Minor.

**Abbreviation**
- RL (Rhythm Loss)
- HTL (Harmonic Tension Loss)
- CLR (Cyclical Learning Rate)
- ER (Early-stop)

**Training details:** 
- Using the config 'multiconditioned_performance_with_dynamics'
- 128 batch size. 
- Drop-out rate 20%
- RL's weight: 0.25
- HTL's weight: 0.25
- CLR's upper-bound and lower-bound at 0.000001 and 0.01. step size was at 180.
- ER's patience was set at 900.

I followed the suggestion from the paper "Early Stopping - But When?" by Lutz Prechelt, which stated that using the Early-stop in the Generalization Loss class was to run the model for several runs and pick the one with the best validation loss. The model that stopped early stopping at around 1,958 steps had the best validation loss.

**Changelog:** 
- The early-stop was activated at around 1,060 steps. The evaluation loss could still be improved. I continued the training.
- The early-stop was activated at around 1,958 steps. The evaluation loss could still be improved. I continued the training.
- The early-stop was activated at around 2,859 steps. The evaluation loss didn't improve anymore.


## List of Modified Files
- **events_rnn_train.py**
    - **Implemented early stopping**
        - Early stopping takes the loss from the evaluation set to determine when the training should stop.
        - You can set the "patience" value.
            - "patience" is the number of steps that the early stopping algorithm will tolerate when it sees that loss from the evaluation set isn't improving anymore.
                - So if you want the algorithm to tolerate 1,000 steps when the loss of the evaluation set isn't improved before stopping the training, you can set the "patience" to 1,000.           
            - The "patience" needs to be set if you want to use early stopping. 
            - To enable Early stopping, the following command line is an example:     
                ```
                --early_stop='patience=1000' \
                ``` 
- **events_rnn_graph.py**
    - **Modified the loss function**
        - Added **Rhythm loss** and **Harmonic Tension Loss** on top of the original loss function.    
            - To enable Rhythm Loss, use the following command line:     
                ```
                --rhythm_loss=0.5 \
                ``` 
            - To enable Harmonic Tension Loss, use the following command line:     
                ```
                --harmony_loss=0.5 \
                ``` 
    - **Added L1 Regularization**
        - Added **L1 Regularization** on top of the original loss function.  
            - To enable L1 Regularization, set the scale value to more than 0. If set to 0, the L1 Regularization won't be used.
            - To enable L1 Regularization, the following command line is an example:     
                ```
                --l1_regular='0.001' \
                ```   
    - **Added Cyclical Learning Rate**  
        - To enable **Cyclical Learning Rate**, set the value for lower-bound, upper-bound, and step size.   
        - The following command line is an example:        
           ```  
           --clr_hparams='lower_bound=0.00025,upper_bound=0.001,step_size=470' \   
           ```     
- **performance_rnn_create_dataset.py**
    - Added a flag to set the compression when creating TFRecord.
       ```  
       performance_rnn_create_dataset \
       --config='multiconditioned_performance_with_dynamics' \
       --input='PATH_TO_YOUR_NOTESEQUENCES.TFRECORD' \
       --output_dir='PATH_TO_YOUR_OUTPUT_DIRECTORY' \
       --eval_ratio=0.1 \
       --use_compression
       ```   

- **pipeline.py**
    - Added compression for TFRecordWriter when the flag is set.

- **data.py**
    - Added a way to handle compressed TFRecord.

- **sequence_example_lib.py**
    - Modified count_records to handle compression.


## Tutorial
I categorized the modified scripts by the original directory it was in the Magenta library.    

To use all the functions I added to the scripts related to Performance RNN, please replace the original scripts with their modified counterparts.   

You can look at this page for the tutorial on how to use the Performance RNN   
https://github.com/magenta/magenta/tree/main/magenta/models/performance_rnn
