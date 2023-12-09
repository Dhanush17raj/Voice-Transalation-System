import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model
from gtts import gTTS
import os
import numpy as np
from sklearn.model_selection import train_test_split
import unicodedata
import re
import io
import time
import sqlite3
from flask import Flask, request, render_template,url_for, redirect,send_file

app = Flask(__name__,static_url_path='/static')
feedback_data = [
    # {"rating": 5, "comment": "Great product!"},
    # {"rating": 4, "comment": "Could be better."},
    # {"rating": 3, "comment": "Average"},
    # {"rating": 2, "comment": "Not up to the mark"},
    # {"rating": 1, "comment": "Poor"}
]


def calculate_average_rating(feedback):
    if not feedback:
        return 0
    total = sum(entry["rating"] for entry in feedback)
    return total / len(feedback)

def initialize_database():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rating INTEGER NOT NULL,
            comment TEXT
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return render_template("audio-only.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        audioFile = request.files['file']

        directory_path = "/home/dhanush/Documents/ML/MinorProject/static/audio"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        file_path = os.path.join(directory_path, audioFile.filename)
        global input_mp3_path
        input_mp3_path=file_path
        audioFile.save(file_path)
        frame_length = 256
        frame_step = 160
        fft_length = 384
        characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]

        # Mapping characters to integers
        char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")

        # Mapping integers back to original characters
        num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
        )
        batch_size = 32
        def decode_batch_predictions(pred):
            input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
            results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        # Iterate over the results and get back the text
            output_text = []
            for result in results:
                result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
                output_text.append(result)
            return output_text
        def encode_single_sample(wav_file):
        # Process the Audio
        # 1. Read wav file
            file = tf.io.read_file(wav_file)
        # 2. Decode the wav file
            audio, _ = tf.audio.decode_wav(file)
        # 3. Squeeze the tensor along the channel axis
            audio = tf.squeeze(audio, axis=-1)
        # 4. Change type to float
            # if len(audio.shape) > 1:
            # # If there's an extra dimension, select the first channel
            #     audio = audio[:, 0]
            audio = tf.cast(audio, tf.float32)
        # 5. Get the spectrogram
            spectrogram = tf.signal.stft(
                audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
            )
        # 6. We only need the magnitude, which can be derived by applying tf.abs
            spectrogram = tf.abs(spectrogram)
            spectrogram = tf.math.pow(spectrogram, 0.5)
        # 7. Normalization
            means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
            stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
            spectrogram = (spectrogram - means) / (stddevs + 1e-10)
            return spectrogram
        def CTCLoss(y_true, y_pred):
        # Compute the training-time loss value
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

            loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
            return loss
        def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
            """Model similar to DeepSpeech2."""
            # Model's input
            input_spectrogram = layers.Input((None, input_dim), name="input")
            # Expand the dimension to use 2D CNN.
            x = layers.Reshape((-1, input_dim, 1), input_shape=(None, input_dim), name="expand_dim")(input_spectrogram)
            # Convolution layer 1
            x = layers.Conv2D(
                filters=32,
                kernel_size=[11, 41],
                strides=[2, 2],
                padding="same",
                use_bias=False,
                name="conv_1",
            )(x)
            x = layers.BatchNormalization(name="conv_1_bn")(x)
            x = layers.ReLU(name="conv_1_relu")(x)
            # Convolution layer 2
            x = layers.Conv2D(
                filters=32,
                kernel_size=[11, 21],
                strides=[1, 2],
                padding="same",
                use_bias=False,
                name="conv_2",
            )(x)
            x = layers.BatchNormalization(name="conv_2_bn")(x)
            x = layers.ReLU(name="conv_2_relu")(x)
            # Reshape the resulted volume to feed the RNNs layers
            x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
            # RNN layers
            for i in range(1, rnn_layers + 1):
                recurrent = layers.GRU(
                    units=rnn_units,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    use_bias=True,
                    return_sequences=True,
                    reset_after=True,
                    name=f"gru_{i}",
                )
                x = layers.Bidirectional(
                    recurrent, name=f"bidirectional_{i}", merge_mode="concat"
                )(x)
                if i < rnn_layers:
                    x = layers.Dropout(rate=0.5)(x)
            # Dense layer
            x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
            x = layers.ReLU(name="dense_1_relu")(x)
            x = layers.Dropout(rate=0.5)(x)
            # Classification layer
            output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
            # Model
            model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
            # Optimizer
            opt = keras.optimizers.Adam(learning_rate=1e-4)
            # Compile the model and return
            model.compile(optimizer=opt, loss=CTCLoss)
            return model

        model1 = load_model('speech_recognition_model.h5', custom_objects={'CTCLoss': CTCLoss})

        wav_file=audioFile

        def decode_batch_predictions(pred, beam_width=5):
            input_len = np.ones(pred.shape[0]) * pred.shape[1]
            # Use beam search instead of greedy search
            results = keras.backend.ctc_decode(pred, input_length=input_len, beam_width=beam_width, top_paths=1)[0][0]
            # Iterate over the results and get back the text
            output_text = []
            for result in results:
                result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
                output_text.append(result)
            return output_text
        X =encode_single_sample(file_path)
        X = tf.expand_dims(X, axis=0)
        batch_predictions = model1.predict(X)
        batch_predictions = decode_batch_predictions(batch_predictions)
        # print(batch_predictions)
        path_to_file ='spa.txt'
        def unicode_to_ascii(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        def preprocess_sentence(w):
            w = unicode_to_ascii(w.lower().strip())
        #w = w.lower().strip()

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
            w = re.sub(r"([?.!,¿])", r" \1 ", w)
            w = re.sub(r'[" "]+', " ", w)

            # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
            #w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

            w = w.lstrip().strip()

            # adding a start and an end token to the sentence
            # so that the model know when to start and stop predicting.
            w = '<start> ' + w + ' <end>'
            return w
        def create_dataset(path, num_examples):
        #lines = io.open('hin.txt', encoding='UTF-8').read().split('\n')
        #lines = lines.strip().split('\n')
        #lines = io.open(path, encoding='UTF-8').readlines().strip().split('\n')
            lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
            word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

            return zip(*word_pairs)
        def max_length(tensor):
            return max(len(t) for t in tensor)
        def tokenize(lang):
            lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
            lang_tokenizer.fit_on_texts(lang)

            tensor = lang_tokenizer.texts_to_sequences(lang)

            tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

            return tensor, lang_tokenizer
        def load_dataset(path, num_examples=None):
        # creating cleaned input, output pairs
            inp_lang, targ_lang, _ = create_dataset(path, num_examples)

            input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
            target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

            return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
        num_examples = 3000
        input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

        # Calculate max_length of the target tensors
        max_length_inp, max_length_targ = max_length(input_tensor), max_length(target_tensor)

        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)
        def convert(lang, tensor):
            for t in tensor:
                if t!=0:
                    print ("%d ----> %s" % (t, lang.index_word[t]))
        BUFFER_SIZE = len(input_tensor_train)
        BATCH_SIZE = 32
        steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
        embedding_dim = 256
        units = 1024
        vocab_inp_size = len(inp_lang.word_index)+1
        vocab_tar_size = len(targ_lang.word_index)+1

        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)          
        example_input_batch, example_target_batch = next(iter(dataset))
        class Encoder(tf.keras.Model):
            def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
                super(Encoder, self).__init__()
                self.batch_sz = batch_sz
                self.enc_units = enc_units
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
                self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True,
                                                return_state=True, recurrent_initializer='glorot_uniform')

            def call(self, x, hidden):
                x = self.embedding(x)
                output, state = self.gru(x, initial_state = hidden)
                return output, state

            def initialize_hidden_state(self):
                return tf.zeros((self.batch_sz, self.enc_units))
        encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
        class BahdanauAttention(tf.keras.Model):
            def __init__(self, units):
                super(BahdanauAttention, self).__init__()
                self.W1 = tf.keras.layers.Dense(units)
                self.W2 = tf.keras.layers.Dense(units)
                self.V = tf.keras.layers.Dense(1)

            def call(self, query, values):
                # hidden shape == (batch_size, hidden size)
                # hidden_with_time_axis shape == (batch_size, 1, hidden size)
                # we are doing this to perform addition to calculate the score
                hidden_with_time_axis = tf.expand_dims(query, 1)

                # score shape == (batch_size, max_length, 1)
                # we get 1 at the last axis because we are applying score to self.V
                # the shape of the tensor before applying self.V is (batch_size, max_length, units)
                score = self.V(tf.nn.tanh(
                    self.W1(values) + self.W2(hidden_with_time_axis)))

                # attention_weights shape == (batch_size, max_length, 1)
                attention_weights = tf.nn.softmax(score, axis=1)

                # context_vector shape after sum == (batch_size, hidden_size)
                context_vector = attention_weights * values
                context_vector = tf.reduce_sum(context_vector, axis=1)
                return context_vector, attention_weights
        class Decoder(tf.keras.Model):
            def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
                super(Decoder, self).__init__()
                self.batch_sz = batch_sz
                self.dec_units = dec_units
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
                self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True,
                                                return_state=True, recurrent_initializer='glorot_uniform')
                self.fc = tf.keras.layers.Dense(vocab_size)

                # used for attention
                self.attention = BahdanauAttention(self.dec_units)

            def call(self, x, hidden, enc_output):
                # enc_output shape == (batch_size, max_length, hidden_size)
                context_vector, attention_weights = self.attention(hidden, enc_output)

                # x shape after passing through embedding == (batch_size, 1, embedding_dim)
                x = self.embedding(x)

                # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
                x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

                # passing the concatenated vector to the GRU
                output, state = self.gru(x)

                # output shape == (batch_size * 1, hidden_size)
                output = tf.reshape(output, (-1, output.shape[2]))

                # output shape == (batch_size, vocab)
                x = self.fc(output)
                return x, state, attention_weights
        
        decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_mean(loss_)
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
        @tf.function
        def train_step(inp, targ, enc_hidden):
            loss  = 0

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, enc_hidden)

                dec_hidden = enc_hidden

                dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                    loss += loss_function(targ[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

            variables = encoder.trainable_variables + decoder.trainable_variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss
        def evaluate(sentence):
            attention_plot = np.zeros((max_length_targ, max_length_inp))

            sentence = preprocess_sentence(sentence)

            inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
            inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
            inputs = tf.convert_to_tensor(inputs)

            result = ''

            hidden = [tf.zeros((1, units))]
            enc_out, enc_hidden = encoder(inputs, hidden)

            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

            for t in range(max_length_targ):
                predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

                # storing the attention weights to plot later on
                attention_weights = tf.reshape(attention_weights, (-1, ))
                attention_plot[t] = attention_weights.numpy()

                predicted_id = tf.argmax(predictions[0]).numpy()

                result += targ_lang.index_word[predicted_id] + ' '

                if targ_lang.index_word[predicted_id] == '<end>':
                    return result, sentence, attention_plot

                # the predicted ID is fed back into the model
                dec_input = tf.expand_dims([predicted_id], 0)

            return result, sentence, attention_plot
        def translate(sentence):
            result, sentence, attention_plot = evaluate(sentence)
            return result
        checkpoint_dir='training_checkpoints'

        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        def filter_words_not_in_dict(sentence, inp_lang):
            words = sentence.split()
            lower_case_word_index = {word.lower() for word in inp_lang.word_index}
            filtered_words = [word for word in words if word.lower() in lower_case_word_index]
            filtered_sentence = ' '.join(filtered_words)
            return filtered_sentence
        filtered_sentence = filter_words_not_in_dict(batch_predictions[0], inp_lang)
        output=translate(filtered_sentence)
                

        def text_to_sound(text, language='es', filename='/home/dhanush/Documents/ML/MinorProject/static/audio/output.mp3'):
            # Create a gTTS object
            tts = gTTS(text=text, lang=language, slow=False)

            # Save the speech as an MP3 file
            tts.save(filename)
        text_to_sound(output)
    arr = 'ENGLISH SENTENCE IS: '+ batch_predictions[0]
    zrr= "TRANSALATION IS: " + output
    return render_template("audio-only.html", pred1=arr,pred2=zrr)
    # return send_file('/home/dhanush/Documents/ML/MinorProject/output.mp3', as_attachment=False)
@app.route('/about')
def about():
    return render_template('about.html')

output_mp3_path = '/home/dhanush/Documents/ML/MinorProject/static/audio/output.mp3'

@app.route('/get_output_audio')
def get_output_audio():
    # Check if the output.mp3 file exists
    if os.path.exists(output_mp3_path):
        # Send the output.mp3 file as a response
        return send_file(output_mp3_path, as_attachment=False)
    else:
        return "File not found"
    
@app.route('/get_input_audio')
def get_input_audio():
    # Check if the output.mp3 file exists
    if os.path.exists(input_mp3_path):
        # Send the output.mp3 file as a response
        return send_file(input_mp3_path, as_attachment=False)
    else:
        return "File not found"
    
# @app.route('/feedback', methods=['GET', 'POST'])
# def feedback():
#     if request.method == 'POST':
#         name= request.form['name']
#         rating = int(request.form['rating'])
#         comment = request.form['comment']

#         # Append new feedback to the list
#         feedback_data.append({"name":name,"rating": rating, "comment": comment})

#     avg_rating = calculate_average_rating(feedback_data)
#     return render_template('feedback.html', feedback_data=feedback_data, avg_rating=avg_rating)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()

    if request.method == 'POST':
        name = request.form['name']
        rating = int(request.form['rating'])
        comment = request.form['comment']

        c.execute('''
            INSERT INTO feedback (name, rating, comment) VALUES (?, ?, ?)
        ''', (name, rating, comment))
        conn.commit()

    c.execute('SELECT name, rating, comment FROM feedback')
    columns = [col[0] for col in c.description]  # Fetch column names
    feedback_data = [dict(zip(columns, row)) for row in c.fetchall()]  # Convert rows to dictionaries

    avg_rating = calculate_average_rating(feedback_data)

    conn.close()
    return render_template('feedback.html', feedback_data=feedback_data, avg_rating=avg_rating)
    
if __name__ == "__main__":
    initialize_database() 
    app.run(debug=True)