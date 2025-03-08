# import huggingface_hub
# import transformers
# from mpmath.libmp import dps_to_prec
# from transformers import TFDistilBertForQuestionAnswering, TrainingArguments, pipeline, AutoTokenizer, TFAutoModelForSequenceClassification, DistilBertTokenizer
# import evaluate

import pandas as pd

import tensorflow as tf

#
# d_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
# dist = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# dist_copy = dist
# # print(distilbert.summary())
#
# for layer in dist.layers[:3]:
#     print(layer.name)
#     print(type(layer.weights[0]))


# Mini test in TF with distilbert
# prompt, info = "The blank should be ",  "All houses are green. If I am a house, then I am  _____."
#
#
# inputs = d_tokenizer(prompt, info, return_tensors="tf")
# outputs = dist(**inputs)
#
# answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
# answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
#
# predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
# answer = d_tokenizer.decode(predict_answer_tokens)
#
# print(answer)


# TRAINING FUNCTION
# def train_model(model):
#     trainings_args = TrainingArguments(output_dir = "dist_trainer")
#     metric = evaluate.load("accuracy")



# LOAD DATA
