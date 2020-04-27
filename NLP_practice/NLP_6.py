import os
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence
print(os.getcwd())
os.chdir('ML_study/PyTorchZeroToAll-master')
device = torch.device("cuda")


if __name__ == '__main__':
    data = ["hello world",
            "midnight",
            "calculation",
            "path",
            "short circuit"]
    char_set = ['<pad>'] + list(set(char for seq in data for char in seq))
    char2idx = {char: idx for idx, char in enumerate(char_set)}
    print('char_set: ', char_set)
    print('char_set length', len(char_set))

    x = [torch.LongTensor([char2idx[char] for char in seq]) for seq in data]
    for sequence in x:
        print(sequence)
    
    lengths = [len(seq) for seq in x]
    print('lengths:', lengths)

    # sequence가 다를 경우?
    # 하나의 batch로 만들어주기 위해서 일반적으로 제일 긴 sequence 길이에 맞춰 뒷부분에 padding을 추가
    # PyTorch의 PackedSequence라는 걸 쓰면 padding 없이도 정확히 필요한 부분까지만 병렬 연산 가능

    # input이 Tensor들의 list로 주어져야 한다.
    # (T, batch_size, a, b, ...) shape을 가지는 Tensor로 리턴됨. (T는 가장긴 squence length)
    # default로 padding은 0이지만 특정 수를 정할 수도 있다.
    padded_sequence = pad_sequence(x, batch_first = True)
    print(padded_sequence)
    print(padded_sequence.shape)

    # pack_sequence 함수를 이용하여 PackedSequence 만들기
    # padding token을 추가하여 sequence의 최대길이에 맞는 tensor를 만드는게 아닌,
    # padding을 추가하지 않고 정확히 주어진 sequence 길이까지만 모델이 연산을 하게끔 만드는 PyTorch의 자료구조이다.
    # PackedSequence를 만들기 위해서는 한가지 조건이 필요하다.
    # 주어진 input(list of Tensor)는 길이에 따른 내림차순 정렬이 되어 있어야 한다.
    sorted_idx = sorted(range(len(lengths)), key=lengths.__getitem__, reverse = True)
    sorted_x = [x[idx] for idx in sorted_idx]
    for sequence in sorted_x:
        print(sequence)

    # Embedding 적용해보기
    # one-hot embedding using PaddedSequence
    eye = torch.eye(len(char_set))
    embedded_tensor = eye[padded_sequence]
    # 5x13x19 sequence 개수 * sequence 길이 * 단어 개수
    print(embedded_tensor.shape)

    embedded_packed_seq = pack_sequence([eye[x[idx]] for idx in sorted_idx])
    print(embedded_packed_seq.data.shape)
    
    # character 개수로 한다.
    rnn = torch.nn.RNN(input_size = len(char_set), hidden_size=30, batch_first = True)
    rnn_output, hidden = rnn(embedded_packed_seq)
    print(rnn_output.data.shape)
    print(hidden.shape)

    # pad_packed_sequence
    # packedSequence를 PaddedSequence(Tensor)로 바꾸어주는 함수
    # PackedSequence는 각 sequence에대한 길이 정보도 가지고 있기 때문에, 이함수는 Tensor와 함게 길이에 대한 리스트를 튜플로 리턴해준다
    # (Tensor, list_of_lenghts)

    unpacked_sequence, seq_lengths = pad_packed_sequence(embedded_packed_seq, batch_first = True)
    print(unpacked_sequence)
    print(seq_lengths)

    # pack_padded_sequence
    # padding된 Tensor인 PaddedSequence를 PackedSequence로 바꾸어주는 함수
    # pack_padded_sequence 함수는 실제 sequence 길이에 대한 정보를 모르기 때문에, 파라미터로 꼭 제공해주어야 한다.
    # 여기서 주의해야할 점은, input인 PaddedSequence가 아까 언급한 **길이에 대한 내림차순으로 정렬 되어야 한다.**
    # 말때부터 정렬해서 말아야 한다.
    embedded_padded_sequence = eye[pad_sequence(sorted_x, batch_first = True)]
    print(embedded_padded_sequence.shape)
    print(embedded_padded_sequence.batch_sizes)
    