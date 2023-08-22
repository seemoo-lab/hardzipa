import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
import os
import sys


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
      #GRU layer is normal RNN layer and can be exchanged with LSTM layer and TODO: check if Transformer layer work as well
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
    ])
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


class TextGenerator:
    def __init__(self,text,path,epoches=10,seq_length=32,batch_size=32,buffer_size=1000,embedding_dim = 256,rnn_units = 1024):
        self.seq_length = seq_length
        self.batch_size=batch_size
        self.buffer_size= buffer_size
        self.epochs=epoches
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.__set_vocab(text)
        self.dataset = self.__create_dataset(self.text_2_int(text))
        self.path = path

        self.model = build_model(len(self.vocab),embedding_dim,rnn_units,batch_size)
        self.model.summary()
        self.model.compile(optimizer="adam",loss=loss)
        self.generate_model = None

    def __set_vocab(self,text):
        self.vocab = sorted(set(text))
        self.char2idx = {u:i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

    def text_2_int(self, text):
        return np.array([self.char2idx[c] for c in text])

    def int_2_txt(self,int):
        return self.idx2char[int]

    def ints_2_text(self, ints):
        return "".join(np.array([self.idx2char[i] for i in ints]))

    def __create_dataset(self,text_as_int):
        charset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = charset.batch(self.seq_length + 1,drop_remainder=True)
        len(list(sequences))
        sequences1 = charset.batch(self.seq_length + 1,drop_remainder=True)
        len(list(sequences1))
        dataset = sequences.map(split_input_target)
        return dataset

    def get_dataset(self):
        return self.dataset

    def train_model(self):
        dataset = self.dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True).repeat(20)
        print(len(list(dataset)))

        # Need to save the Cehckpoints in order to create a model with only one input and load the saved weights :D
        checkpoint_dir = os.path.join(self.path,'training_checkpoints')
        log_dir = os.path.join(self.path,'log')
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.fit(dataset,epochs=self.epochs, callbacks=[checkpoint_callback,tensorboard_callback])

        #now change model to take only one input
        model = build_model(len(self.vocab), self.embedding_dim, self.rnn_units, batch_size=1)

        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

        model.build(tf.TensorShape([1, None]))
        self.generate_model = model

    def save_model(self):
        path = os.path.join(self.path,"saved_mode","textgen_model")
        os.makedirs(path,exist_ok=True)
        self.generate_model.save(path)

    def load_model(self, path=os.path.join("saved_mode","textgen_model")):
        self.generate_model = tf.keras.models.load_model(os.path.join(self.path,path))

    def default_pretrained_exists(self):
        path = os.path.join("saved_mode","textgen_model")
        return os.path.exists(os.path.join(self.path,path))

    def generate_text(self,start_string="\n",temp = 0.5): #TODO: start_string should be chosen random from vocabulary
        # Number of characters to generate
        if self.generate_model is None:
            return
        stopchar = "\n"
        if start_string is None:
            rand = int.from_bytes(os.urandom(32), sys.byteorder) % len(self.vocab)
            start_string = self.vocab[rand]

        # Converting our start string to numbers (vectorizing)
        input_eval = self.text_2_int(start_string)
        input_eval = tf.expand_dims(input_eval, 0)
        input_eval = tf.cast(input_eval,tf.float32)
        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = temp

        # Here batch size == 1
        self.generate_model.reset_states()
        while True:
            predictions = self.generate_model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
            char = self.int_2_txt(predicted_id)
            if char != stopchar:
                text_generated.append(char)
            else:
                return start_string + ''.join(text_generated)
