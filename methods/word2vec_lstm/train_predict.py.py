#!/usr/bin/env python
# coding: utf-8

# # MBTI Prediction: Word2Vec + Bi-LSTM (GPU Optimized)
# 
# This notebook implements MBTI personality prediction using:
# - **Embedding**: Word2Vec (trained on corpus)
# - **Model**: Bidirectional LSTM
# - **Task**: 4 binary classifications (E/I, N/S, T/F, P/J)
# - **Optimization**: NVIDIA RTX 3090 (24GB VRAM)

# ## 1. GPU Setup and Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

# Word2Vec
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt', quiet=True)

# GPU Configuration for RTX 3090
print("Setting up GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Enable mixed precision for faster training
        mixed_precision.set_global_policy('mixed_float16')
        
        print(f"GPU Available: {len(gpus)} GPU(s)")
        print(f"GPU Name: {gpus[0].name}")
        print(f"Mixed Precision: Enabled (float16)")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU")

print(f"\nTensorFlow version: {tf.__version__}")


# ## 2. Load Data

# In[ ]:


# Load data from GitHub
TRAIN_URL = 'https://raw.githubusercontent.com/beefed-up-geek/nlp_final_project/refs/heads/main/kaggle_data/2025MBTItrain.csv'
TEST_URL = 'https://raw.githubusercontent.com/beefed-up-geek/nlp_final_project/refs/heads/main/kaggle_data/2025test.csv'

print("Loading data...")
train_df = pd.read_csv(TRAIN_URL)
test_df = pd.read_csv(TEST_URL)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nMBTI type distribution:")
print(train_df['type'].value_counts().head(10))


# ## 3. Text Preprocessing

# In[ ]:


def preprocess_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    text = ' '.join(text.split())
    return text

print("Preprocessing texts...")
train_df['cleaned_posts'] = train_df['posts'].apply(preprocess_text)
test_df['cleaned_posts'] = test_df['posts'].apply(preprocess_text)
print("Preprocessing complete!")


# ## 4. Create Binary Labels

# In[ ]:


train_df['E_I'] = train_df['type'].apply(lambda x: 1 if x[0] == 'E' else 0)
train_df['N_S'] = train_df['type'].apply(lambda x: 1 if x[1] == 'N' else 0)
train_df['T_F'] = train_df['type'].apply(lambda x: 1 if x[2] == 'T' else 0)
train_df['P_J'] = train_df['type'].apply(lambda x: 1 if x[3] == 'P' else 0)

print("Label distribution:")
for col in ['E_I', 'N_S', 'T_F', 'P_J']:
    dist = train_df[col].value_counts()
    print(f"{col}: 0={dist[0]}, 1={dist[1]} (ratio: {dist[1]/len(train_df):.2%})")


# ## 5. Tokenization

# In[ ]:


# Parameters optimized for 3090 GPU
MAX_WORDS = 20000
MAX_SEQUENCE_LENGTH = 400

print("Tokenizing texts...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['cleaned_posts'])

X_train_seq = tokenizer.texts_to_sequences(train_df['cleaned_posts'])
X_test_seq = tokenizer.texts_to_sequences(test_df['cleaned_posts'])

X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

print(f"Vocabulary size: {len(tokenizer.word_index)}")
print(f"Train sequences shape: {X_train_padded.shape}")
print(f"Test sequences shape: {X_test_padded.shape}")


# ## 6. Train Word2Vec Embeddings

# In[ ]:


print("Training Word2Vec model...")
all_texts = pd.concat([train_df['cleaned_posts'], test_df['cleaned_posts']])
tokenized_texts = [word_tokenize(text) for text in all_texts]

EMBEDDING_DIM = 128
w2v_model = Word2Vec(
    sentences=tokenized_texts,
    vector_size=EMBEDDING_DIM,
    window=5,
    min_count=2,
    workers=8,  # Use multiple CPU cores
    sg=1,
    epochs=10
)

print(f"Word2Vec trained: {len(w2v_model.wv)} words")

# Create embedding matrix
vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM), dtype=np.float32)

for word, idx in tokenizer.word_index.items():
    if idx >= MAX_WORDS:
        continue
    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]
    else:
        embedding_matrix[idx] = np.random.normal(0, 0.05, EMBEDDING_DIM)

print(f"Embedding matrix shape: {embedding_matrix.shape}")
print(f"Coverage: {np.count_nonzero(embedding_matrix.sum(axis=1))} / {vocab_size}")


# ## 7. Build Model

# In[ ]:


def create_bilstm_model(embedding_matrix, max_length):
    """GPU-optimized Bi-LSTM model"""
    input_layer = Input(shape=(max_length,))
    
    embedding = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=True
    )(input_layer)
    
    # Bi-LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output (float32 for mixed precision)
    output = Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ## 8. Train Models

# In[ ]:


# Training configuration for 3090 GPU
BATCH_SIZE = 64  # Optimized for 24GB VRAM
EPOCHS = 50
PATIENCE = 5
VALIDATION_SPLIT = 0.2

models = {}
histories = {}
dimensions = ['E_I', 'N_S', 'T_F', 'P_J']

for dim in dimensions:
    print(f"\n{'='*60}")
    print(f"Training {dim} classifier")
    print(f"{'='*60}")
    
    y = train_df[dim].values
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_padded, y, 
        test_size=VALIDATION_SPLIT, 
        random_state=42,
        stratify=y
    )
    
    model = create_bilstm_model(embedding_matrix, MAX_SEQUENCE_LENGTH)
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_tr, y_tr,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n✓ Best Validation Accuracy: {val_acc:.4f}")
    
    models[dim] = model
    histories[dim] = history
    
    # Clear memory
    tf.keras.backend.clear_session()
    
print("\n" + "="*60)
print("All models trained!")
print("="*60)


# ## 9. Generate Predictions

# In[ ]:


print("Generating predictions...")
predictions = {}

for dim in dimensions:
    pred_proba = models[dim].predict(X_test_padded, batch_size=BATCH_SIZE, verbose=0)
    pred_binary = (pred_proba > 0.5).astype(int).flatten()
    predictions[dim] = pred_binary
    print(f"{dim}: {np.bincount(pred_binary)}")

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'E_I': predictions['E_I'],
    'N_S': predictions['N_S'],
    'T_F': predictions['T_F'],
    'P_J': predictions['P_J']
})

print(f"\nSubmission shape: {submission.shape}")
print(submission.head(10))


# ## 10. Save Results

# In[ ]:


filename = 'submission_word2vec_lstm.csv'
submission.to_csv(filename, index=False)
print(f"✓ Submission saved: {filename}")

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
for dim in dimensions:
    best_acc = max(histories[dim].history['val_accuracy'])
    print(f"{dim}: {best_acc:.4f}")
print("="*60)

