import numpy as np
from bones.helpers import to_one_hot


def indices_to_data(indices, seqlength, num_chars):
    num_examples = len(indices) - seqlength
    x = np.zeros([num_examples, seqlength], dtype='float32')
    y = np.zeros([num_examples], dtype='int32')
    for example_num in range(0, len(indices) - seqlength):
        start = example_num
        end = start + seqlength
        x[example_num, :] = indices[start:end]
        y[example_num] = indices[end]
    return to_one_hot(x, num_chars), to_one_hot(y, num_chars)


def load_text(text_fnm, seqlength=20):
    with open(text_fnm) as f:
        text = f.read()
    chars = list(set(text))
    char_to_index = {ch: i for i, ch in enumerate(chars)}
    index_to_char = {i: ch for i, ch in enumerate(chars)}
    x, y = indices_to_data([char_to_index[c] for c in text],
                           seqlength, len(chars))
    return x, y, {'char_to_index': char_to_index,
                  'index_to_char': index_to_char,
                  'seqlength': seqlength}


def print_fn(dicts, num_chars=100, seed='The quick brown fox jumps'):
    def fn(step, epoch, loss, model):
        print('Loss at step {:d}, epoch {:d}: {:0.4f}'
              .format(step, epoch, loss))
        seq = gen_sequence(model, dicts['seqlength'],
                           dicts['char_to_index'], dicts['index_to_char'],
                           seed, num_chars)
        print('Generated: {}'.format(seq))
    return fn


def gen_sequence(model, seqlength, char_to_index, index_to_char,
                 seed='The quick brown fox jumps', num_chars=100):
    assert len(seed) >= seqlength
    samples = []
    indices = [char_to_index[c] for c in seed]
    data = np.zeros((1, seqlength))
    for i, index in enumerate(indices[:seqlength]):
        data[0, i] = index
    data = to_one_hot(data, len(char_to_index))

    for i in range(num_chars):
        # Pick the character that got assigned the highest probability
        preds = model(data)
        # ix = np.argmax(preds.ravel())
        # Alternatively, to sample from the distribution instead:
        ix = np.random.choice(np.arange(len(char_to_index)), p=preds.ravel())
        samples.append(ix)
        data[0, 0:seqlength - 1] = data[0, 1:]  # bump down
        data[0, seqlength - 1] = ix  # insert latest
    random_snippet = seed + ''.join(
        index_to_char[index] for index in samples)
    return random_snippet
