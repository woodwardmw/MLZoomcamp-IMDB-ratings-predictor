from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertConfig
from transformers import AutoTokenizer, DataCollatorWithPadding



def config_model(checkpoint, params = None):
    if params:
        dropout = params['layer_dropout']
        attention_dropout=params['att_dropout']
    else:
        dropout = 0
        attention_dropout = 0
 
    # Configure DistilBERT's initialization
    config = DistilBertConfig(dropout=dropout , 
                              attention_dropout=attention_dropout, 
                              output_hidden_states=True)

    # The bare, pre-trained DistilBERT transformer model outputting raw hidden-states 
    # and without any specific head on top.
    distilBERT = TFDistilBertModel.from_pretrained(checkpoint, config=config)

    # Make DistilBERT layers untrainable
    for layer in distilBERT.layers:
        layer.trainable = False
    return distilBERT

def build_model(transformer, params = None):
    if params:
        max_length = params['max_length']
        learning_rate=params['lr']
    else:
        max_length = 512
        learning_rate = 0
    """
    Template for building a model off of the BERT or DistilBERT architecture
    for a binary classification task.
    
    Input:
      - transformer:  a base Hugging Face transformer model object (BERT or DistilBERT)
                      with no added classification head attached.
      - max_length:   integer controlling the maximum number of encoded tokens 
                      in a given sequence.
    
    Output:
      - model:        a compiled tf.keras.Model with added classification layers 
                      on top of the base pre-trained model architecture.
    """
    
    # Define weight initializer with a random seed to ensure reproducibility
    weight_initializer = tf.keras.initializers.GlorotNormal() 
    
    # Define input layers
    input_ids_layer = tf.keras.layers.Input(shape=(max_length,), 
                                            name='input_ids', 
                                            dtype='int32')
    input_attention_layer = tf.keras.layers.Input(shape=(max_length,), 
                                                  name='attention_mask', 
                                                  dtype='int32')
    
    # DistilBERT outputs a tuple where the first element at index 0
    # represents the hidden-state at the output of the model's last layer.
    # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
    last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]
    
    # We only care about DistilBERT's output for the [CLS] token, 
    # which is located at index 0 of every encoded sequence.  
    # Splicing out the [CLS] tokens gives us 2D data.
    # cls_token = last_hidden_state[:, 0, :]
    
    ##                                                 ##
    ## Define additional dropout and dense layers here ##
    ##                                            
    
    # vector = tf.reshape(last_hidden_state, [-1])

    vector = tf.keras.layers.GlobalAvgPool1D()(last_hidden_state)

    output1 = tf.keras.layers.Dense(128, 
                                   activation='relu',
                                   kernel_initializer=weight_initializer,  
                                   kernel_constraint=None,
                                   bias_initializer='zeros'
                                   )(vector)
    
    output2 = tf.keras.layers.Dense(128, 
                                   activation='relu',
                                   kernel_initializer=weight_initializer,  
                                   kernel_constraint=None,
                                   bias_initializer='zeros'
                                   )(output1)
    
    output3 = tf.keras.layers.Dense(64, 
                                   activation='relu',
                                   kernel_initializer=weight_initializer,  
                                   kernel_constraint=None,
                                   bias_initializer='zeros'
                                   )(output2)

    output4 = tf.keras.layers.Dense(10, 
                                   activation='relu',
                                   kernel_initializer=weight_initializer,  
                                   kernel_constraint=None,
                                   bias_initializer='zeros'
                                   )(output3)
    
    # Define a single node that makes up the output layer (for binary classification)
    output5 = tf.keras.layers.Dense(1, 
                                   activation=None,
                                   kernel_initializer=weight_initializer,  
                                   kernel_constraint=None,
                                   bias_initializer='zeros'
                                   )(output4)
    
    # Define the model
    model = tf.keras.Model([input_ids_layer, input_attention_layer], output5)
    
    # Compile the model
    model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss=MeanSquaredError(),
                  metrics=['mse'])
    
    return model