import tensorflow as tf

# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device='cpu'
eval_iters = 200
n_embed = 32
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

class FeedForward(tf.keras.Model):
    def __init__(self, n_embed):
        super().__init__()
        self.linear_1 = tf.keras.layers.Dense(input_shape=(n_embed, ), units=4*n_embed, activation='relu')
        self.linear_2 = tf.keras.layers.Dense(units = n_embed)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class Block(tf.keras.Model):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = tf.keras.layers.MultiHeadAttention(
            n_head,
            head_size,
        )
        self.ffwd = FeedForward(n_embed)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        x = x + self.sa(self.ln1(x), self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = tf.keras.layers.Embedding(vocab_size, n_embed)
        self.position_embedding_table = tf.keras.layers.Embedding(block_size, n_embed)
        self.blocks = tf.keras.Sequential(
            [Block(n_embed=n_embed, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = tf.keras.layers.LayerNormalization()
        self.lm_head = tf.keras.layers.Dense(units=vocab_size)

    def call(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(tf.range(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
data = tf.convert_to_tensor(encode(text))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = tf.random.uniform(shape=(batch_size,), maxval=len(data) - block_size, dtype=tf.int32)
    x = tf.stack([tf.slice(data, [i], [block_size]) for i in ix])
    y = tf.stack([tf.slice(data, [i+1], [block_size]) for i in ix])
    return x, y


def get_tf_dataset(data):
    indices = tf.range(tf.shape(data)[0] - block_size)
    dataset = tf.data.Dataset.from_tensor_slices(indices)
    dataset = dataset.map(extract_data_and_target)
    return dataset

def extract_data_and_target(index):
    # Assume that the data is the value at the index and the target is the next value
    x = tf.slice(data, [index], [block_size])
    y = tf.slice(data, [index+1], [block_size])
    return x, y

indices = tf.range(tf.shape(data)[0] - block_size)
dataset = tf.data.Dataset.from_tensor_slices(indices)
dataset = dataset.map(extract_data_and_target)
# Batch your dataset
batch_size = 4
dataset = dataset.batch(batch_size)

model = BigramLanguageModel()
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy()
)
history = model.fit(
    get_tf_dataset(train_data),
    epochs=5,
    steps_per_epoch=1000,
    validation_data=get_tf_dataset(val_data),
    batch_size=batch_size
)

