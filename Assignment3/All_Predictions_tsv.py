for seq_index in range(3000):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    input_pre.append(input_texts[seq_index])
    prediction.append(decoded_sentence)
    
import csv

with open('/content/drive/MyDrive/Colab Notebooks/predictions.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in range(3000):
      tsv_writer.writerow([input_pre[i], prediction[i]])    
