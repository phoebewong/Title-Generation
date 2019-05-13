#rnn_model.py
#Karina Huang
#all models explored for project
#code adapted from computefest

#getBaseModel - Encoder(BI-LSTM)+Decoder(LSTM)
#getBidirectionalModel - Encoder(Bi-LSTM)+Decoder(Bi-LSTM)
#getAttentionModel - Encoder(Bi-LSTM)+Decoder(Bi-LSTM)+Attention
#getAttentionLSTMModel = Encoder(Bi-LSTM)+Decoder(LSTM)+Attention


#import dependencies
from tensorflow.contrib import keras
from keras.layers import Bidirectional, Dropout, Dense,LSTM,Input,Activation,Add,TimeDistributed,\
Permute,Flatten,RepeatVector,merge,Lambda,Multiply,Reshape, Concatenate, Dot
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.models import Sequential,Model
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import tensorflow as tf

#params for model training
#maximum length for title
# tMaxLen = 20
tMaxLen = 250
#maximum length for abstract
aMaxLen = 250
#total maximum length
maxlen = tMaxLen + aMaxLen


def getBaseModel(genTrain, genVal, embeddMatrix,
                 learning_rate, clip_norm, nUnique,
                 embeddDim, hidden_units, encoder_shape = aMaxLen,
                 decoder_shape = tMaxLen):

    '''Base Model'''

    #ENCODER
    #input shape as the vector of sequence, with length padded to 250
    encoder_inputs = Input(shape = (encoder_shape, ), name = 'encoder_input')

    #encode input with embedding layer
    encoder_embedding = Embedding(nUnique, embeddDim,
                                  input_length = encoder_shape,
                                  weights = [embeddMatrix],
                                  mask_zero = True,
                                  name = 'encoder_embedd')(encoder_inputs)

    #1-layer bidirectional LSTM
    encoder_lstm = Bidirectional(LSTM(hidden_units, dropout_U = 0.2,
                                      dropout_W = 0.2 , return_state=True))

    #get states from Bi-LSTM
    encoder_outputs, f_h, f_c, b_h, b_c = encoder_lstm(encoder_embedding)

    #add final states together
    state_hfinal=Add()([f_h, b_h])
    state_cfinal=Add()([f_c, b_c])

    #save encoder states
    encoder_states = [state_hfinal,state_cfinal]

    #DECODER
    decoder_inputs = Input(shape = (decoder_shape, ), name = 'decoder_input')

    #encode decoder input with embedding matrix
    decoder_embedding = Embedding(nUnique, embeddDim,
                                  input_length = decoder_shape,
                                  weights = [embeddMatrix],
                                  mask_zero = True,
                                  name = 'decoder_embedd')

    #1-layer lstm
    decoder_lstm = LSTM(hidden_units,return_sequences = True, return_state=True)

    #save decoder outputs
    decoder_outputs, s_h, s_c = decoder_lstm(decoder_embedding(decoder_inputs), initial_state = encoder_states)
    # decoder_dense = Dense(decoder_shape, activation='linear')

    #time distributed layer, probability predictions for all unique words
    decoder_time_distributed = TimeDistributed(Dense(nUnique,
                                                     name = 'decoder_timedistributed'))
    decoder_activation = Activation('softmax', name = 'decoder_activation')
    decoder_outputs = decoder_activation(decoder_time_distributed(decoder_outputs))

    #MODEL
    model = Model(inputs = [encoder_inputs,decoder_inputs], outputs = decoder_outputs)
    rmsprop = RMSprop(lr = learning_rate, clipnorm = clip_norm)
    model.compile(loss = 'categorical_crossentropy', optimizer = rmsprop)
    return model


def getBidirectionalModel(genTrain, genVal, embeddMatrix,
                          learning_rate, clip_norm, nUnique,
                          embeddDim, hidden_units, encoder_shape = aMaxLen,
                          decoder_shape = tMaxLen):

    '''Bidirectional Model'''

    #ENCODER
    #input shape as the vector of sequence, with length padded to 250
    encoder_inputs = Input(shape = (encoder_shape, ), name = 'encoder_input')

    #encode input with embedding layer
    encoder_embedding = Embedding(nUnique, embeddDim,
                                  input_length = encoder_shape,
                                  weights = [embeddMatrix],
                                  mask_zero = True,
                                  name = 'encoder_embedd')(encoder_inputs)

    #1-layer bidirectional LSTM
    encoder_lstm = Bidirectional(LSTM(hidden_units, dropout_U = 0.2,
                                      dropout_W = 0.2 , return_state=True))

    #get states from Bi-LSTM
    encoder_outputs, f_h, f_c, b_h, b_c = encoder_lstm(encoder_embedding)

    #save encoder states
    encoder_states = [f_h, f_c, b_h, b_c]

    #DECODER
    decoder_inputs = Input(shape = (decoder_shape, ), name = 'decoder_input')

    #encode decoder input with embedding matrix
    decoder_embedding = Embedding(nUnique, embeddDim,
                                  input_length = decoder_shape,
                                  weights = [embeddMatrix],
                                  mask_zero = True,
                                  name = 'decoder_embedd')
    
    #1-layer bidirectional LSTM
    decoder_lstm = Bidirectional(LSTM(hidden_units, dropout_U = 0.2,
                                      dropout_W = 0.2, 
                                      return_sequences = True,
                                      return_state=True))

    #save decoder outputs
    decoder_outputs, f_h, f_c, b_h, b_c = decoder_lstm(decoder_embedding(decoder_inputs), 
                                                       initial_state = encoder_states)

    #time distributed layer, probability predictions for all unique words
    decoder_time_distributed = TimeDistributed(Dense(nUnique,
                                                     name = 'decoder_timedistributed'))
    decoder_activation = Activation('softmax', name = 'decoder_activation')
    decoder_outputs = decoder_activation(decoder_time_distributed(decoder_outputs))

    #MODEL
    model = Model(inputs = [encoder_inputs,decoder_inputs], outputs = decoder_outputs)
    rmsprop = RMSprop(lr = learning_rate, clipnorm = clip_norm)
    model.compile(loss = 'categorical_crossentropy',optimizer = rmsprop)
    return model


def getAttentionModel(genTrain, genVal, embeddMatrix,
                      learning_rate, clip_norm, nUnique,
                      embeddDim, hidden_units, encoder_shape = aMaxLen,
                      decoder_shape = tMaxLen):

    '''Attention/Context Mechanism with Bi-LSTM.'''

    #ENCODER
    #input shape as the vector of sequence, with length padded to 250
    encoder_inputs = Input(shape = (encoder_shape, ), name = 'encoder_input')

    #encode input with embedding layer
    #do not mask 0s because the attention layer does not allow this
    encoder_embedding = Embedding(nUnique, embeddDim,
                                  input_length = encoder_shape,
                                  weights = [embeddMatrix],
                                  mask_zero = True,
                                  name = 'encoder_embedd')(encoder_inputs)

    #1-layer bidirectional LSTM
    encoder_lstm = Bidirectional(LSTM(hidden_units, dropout_U = 0.2, dropout_W = 0.2, 
                                      return_sequences = True, return_state=True))

    #get states from Bi-LSTM
    encoder_outputs, f_h, f_c, b_h, b_c = encoder_lstm(encoder_embedding)

    #save encoder states
    encoder_states = [f_h, f_c, b_h, b_c]

    #DECODER
    decoder_inputs = Input(shape = (decoder_shape, ), name = 'decoder_input')

    #encode decoder input with embedding matrix
    decoder_embedding = Embedding(nUnique, embeddDim,
                                  input_length = decoder_shape,
                                  weights = [embeddMatrix],
                                  mask_zero = True,
                                  name = 'decoder_embedd')

    #1-layer bidirectional lstm
    decoder_lstm = Bidirectional(LSTM(hidden_units, dropout_U = 0.2, dropout_W = 0.2, 
                                      return_sequences = True, return_state=True))

    #save decoder outputs
    decoder_outputs, f_h, f_c, b_h, b_c = decoder_lstm(decoder_embedding(decoder_inputs), 
                                                       initial_state = encoder_states)
  
    #ATTENTION
    attention = Dot(axes = [2,2])([encoder_outputs, decoder_outputs])
    attention = Activation('softmax')(attention)
    context = Dot(axes = [2,1])([attention, encoder_outputs])
    decoder_combined_context = Concatenate()([context, decoder_outputs])


    #time distributed layer, probability predictions for all unique words
    decoder_time_distributed = TimeDistributed(Dense(nUnique,
                                                     name = 'decoder_timedistributed'))
    decoder_activation = Activation('softmax', name = 'decoder_activation')
    decoder_outputs = decoder_activation(decoder_time_distributed(decoder_combined_context))

    #MODEL
    model = Model(inputs = [encoder_inputs,decoder_inputs], outputs = decoder_outputs)
    rmsprop = RMSprop(lr = learning_rate, clipnorm = clip_norm)
    model.compile(loss = 'categorical_crossentropy',optimizer = rmsprop)
    return model


def getAttentionLSTMModel(genTrain, genVal, embeddMatrix,
                          learning_rate, clip_norm, nUnique,
                          embeddDim, hidden_units, encoder_shape = aMaxLen,
                          decoder_shape = tMaxLen):

    '''
    Model with Attention/Context Mechanism with LSTM.
    Attention model code @ https://github.com/wanasit/katakana.git
    '''

    #ENCODER
    #input shape as the vector of sequence, with length padded to 250
    encoder_inputs = Input(shape = (encoder_shape, ), name = 'encoder_input')

    #encode input with embedding layer
    #do not mask 0s because the attention layer does not allow this
    encoder_embedding = Embedding(nUnique, embeddDim,
                                  input_length = encoder_shape,
                                  weights = [embeddMatrix],
                                  mask_zero = True,
                                  name = 'encoder_embedd')(encoder_inputs)

    #forward
    encoder_lstm = LSTM(hidden_units, dropout_U = 0.2, dropout_W = 0.2, 
                        return_sequences = True, return_state=True)
    encoder_lstm_rev = LSTM(hidden_units, dropout_U = 0.2, dropout_W = 0.2,
                            go_backwards = True,
                            return_sequences = True, return_state=True)

    #get states from LSTM
    encoder_outputs_f, h_f, c_f = encoder_lstm(encoder_embedding)
    encoder_outputs_r, h_r, c_r = encoder_lstm_rev(encoder_embedding)

    #save encoder states
    state_hfinal=Add()([h_f, h_r])
    state_cfinal=Add()([c_f, c_r])

    #save encoder states
    encoder_states = [state_hfinal,state_cfinal]

    #DECODER
    decoder_inputs = Input(shape = (decoder_shape, ), name = 'decoder_input')

    #encode decoder input with embedding matrix
    decoder_embedding = Embedding(nUnique, embeddDim,
                                  input_length = decoder_shape,
                                  weights = [embeddMatrix],
                                  mask_zero = True,
                                  name = 'decoder_embedd')
    
    #1-layer lstm
    decoder_lstm = LSTM(hidden_units,return_sequences = True, return_state=True)

    #save decoder outputs
    decoder_outputs, s_h, s_c = decoder_lstm(decoder_embedding(decoder_inputs), initial_state = encoder_states)
  
    #ATTENTION
    attention = Dot(axes = [2,2], name = 'attention')([decoder_outputs, encoder_outputs_f])
    attention = Activation('softmax')(attention)
    context = Dot(axes = [2,1], name = 'context')([attention, encoder_outputs_f])
    decoder_combined_context = Concatenate()([context, decoder_outputs])


    #time distributed layer, probability predictions for all unique words
    decoder_time_distributed = TimeDistributed(Dense(nUnique,
                                                     name = 'decoder_timedistributed'))
    decoder_activation = Activation('softmax', name = 'decoder_activation')
    decoder_outputs = decoder_activation(decoder_time_distributed(decoder_combined_context))

    #MODEL
    model = Model(inputs = [encoder_inputs,decoder_inputs], outputs = decoder_outputs)
    rmsprop = RMSprop(lr = learning_rate, clipnorm = clip_norm)
    model.compile(loss = 'categorical_crossentropy',optimizer = rmsprop)
    return model
