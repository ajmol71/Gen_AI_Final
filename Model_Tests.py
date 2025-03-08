import huggingface_hub
import transformers
from mpmath.libmp import dps_to_prec
from transformers import (TFDistilBertForQuestionAnswering, Trainer, TrainingArguments,
                          pipeline, AutoTokenizer, TFAutoModelForSequenceClassification,
                          DistilBertTokenizer)
import evaluate

import numpy as np
import pandas as pd
import csv
import random

import tensorflow as tf
import keras
import tf_keras

import sklearn
from sklearn.model_selection import train_test_split

# Get Model Weights
# dist_copy = dist
# # print(distilbert.summary())
#
# for layer in dist.layers[:3]:
#     print(layer.name)
#     print(type(layer.weights[0]))


# ----------- Methods ----------- #
# TEST MODEL
def test_model(tokenizer, model, test_data):
    rows_list = []
    for index, row in test_data.iterrows():
        question, prompt = row["question"], row["prompt"]

        full_question = f" Here are the options: A. {row["choice_a"]}  B. {row["choice_b"]}, C. {row["choice_c"]}, D. {row["choice_d"]}, E. {row["choice_e"]}" + question

        inputs = tokenizer(full_question, prompt, return_tensors="tf")
        outputs = model(**inputs)

        answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
        answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        answer = tokenizer.decode(predict_answer_tokens)

        if row['correct_a'] in answer:
            correct = 1
        else:
            correct = 0

        print("\nQUESTION: ", row['question'])
        print("ANSWER: ", answer)

        q_dict = {"model": model.name, "q_num": row["q_num"], "q_type": row["q_type"], "answer": answer, "correct": correct}

        rows_list.append(q_dict)
    pd_df = pd.DataFrame(rows_list)
    return pd_df

def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# TRAINING FUNCTION
def train_model(model, train_data):
    training_args = TrainingArguments(output_dir = model.name + "_trainer")
    metric = evaluate.load("accuracy")

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=train_data,
        compute_metrics=compute_metrics,
    )

    return trainer.train()


# -------- LOAD DATA -------- #
data_header = ["model", "q_num", "q_type", "response", "correct"]

and_qs = pd.read_csv("AnD_questions.csv")
lsat_qs = pd.read_csv("LSAT_Questions.csv")

lsat_LR = lsat_qs[lsat_qs["q_type"]=="LR"]
lsat_LP = lsat_qs[lsat_qs["q_type"]=="LP"]

lsat_train_LR, lsat_test_LR = train_test_split(lsat_LR, train_size = 50/65, test_size = 15/65, random_state=0)
lsat_train_LP, lsat_test_LP = train_test_split(lsat_LP, train_size = 50/65, test_size = 15/65, random_state=0)

lsat_train = pd.concat([lsat_train_LR, lsat_train_LP])
lsat_test = pd.concat([lsat_test_LR, lsat_test_LP])



# --------- MODEL: DISTILBERT --------- #
d_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
dist = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# answer_df = test_model(d_tokenizer, dist, and_qs)
#
# answer_df.to_csv("AnD_qs_performance_2.csv")
#
# trained_model = train_model(dist, )
#
# answer_df_2 = test_model()

answer_df = test_model(d_tokenizer, dist, lsat_test)
answer_df.to_csv("LSAT_qs_performance.csv")

print()
print(answer_df.head())


