import huggingface_hub
import transformers
from mpmath.libmp import dps_to_prec
from transformers import (AutoModelForCausalLM, AutoModelForQuestionAnswering, Trainer, TrainingArguments,
                          pipeline, AutoTokenizer, AutoModelForMaskedLM, TFAutoModelForCausalLM, TFGPT2Tokenizer)
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import encodings
import outcome


# ----------- Methods ----------- #
def load_model(m_name):
    # model = AutoModelForCausalLM.from_pretrained(m_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(m_name)
    tokenizer = AutoTokenizer.from_pretrained(m_name)
    return model, tokenizer

def create_full_q(data):
    dataframe = data.copy()
    for index, row in data.iterrows():
        full_question = row['prompt'] + f" Here are the options: A. {row["choice_a"]}  B. {row["choice_b"]}, C. {row["choice_c"]}, D. {row["choice_d"]}, E. {row["choice_e"]}" + row['question'] + f". The answer is {row['label']}."
        dataframe["Full_Q"] = full_question

    return dataframe


def test_model(tokenizer, model, m_name, test_data, pytf):
    rows_list = []
    count = 0
    for index, row in test_data.iterrows():
        question, prompt = row["question"], row["prompt"]

        full_question = f" Here are the options: A. {row["choice_a"]}  B. {row["choice_b"]}, C. {row["choice_c"]}, D. {row["choice_d"]}, E. {row["choice_e"]}" + question

        inputs = tokenizer(full_question, prompt, return_tensors=pytf)
        outputs = model(**inputs)

        if pytf == "tf":
            answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
            answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
            predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
            answer = tokenizer.decode(predict_answer_tokens)
        elif pytf != "pt":
            all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
            answer_tokens = all_tokens[torch.argmax(outputs["start_logits"]):torch.argmax(outputs["end_logits"]) + 1]
            answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
        else:
            index_list = []
            for i in range(outputs['logits'][0].shape[0]):
                index = torch.argmax(outputs['logits'][0][i])
                index_list.append(int(index))
            indices = torch.tensor(index_list)
            answer = tokenizer.decode(indices)

        if row['correct_a'] in answer:
            correct = 1
        else:
            correct = 0

        print("\nQUESTION: ", row['question'])
        print("ANSWER: ", answer)

        q_dict = {"model": m_name, "q_num": row["q_num"], "q_type": row["q_type"], "answer": answer, "correct": correct}

        rows_list.append(q_dict)
    pd_df = pd.DataFrame(rows_list)
    return pd_df

def print_mdl_weights(model):
    print(model.summary())

    for layer in model.layers:
        print(layer.name)
        print(type(layer.weights[0]))
        print(layer.weights)
    return

def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_model(model, m_name):
    training_args = TrainingArguments(output_dir = m_name + "_trainer")
    metric = evaluate.load("accuracy")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lsat_train_dataset,
        eval_dataset=lsat_valid_dataset,
        compute_metrics=compute_metrics,
    )
    return trainer.train()

def append_test_log(file_name, new_data):
    with open(file_name, 'a') as file:
        writer = csv.writer(file)

        for index, row in new_data.iterrows():
            writer.writerow(row)
    return

def complete_tests(model, tokenizer, m_name, pytf="pt"):
    # AnD Tests
    answer_df = test_model(tokenizer, model, m_name, and_qs, pytf)
    append_test_log("AnD_qs_performance_3.csv", answer_df)

    # LSAT Tests
    answer_df = test_model(tokenizer, model, m_name, lsat_test, pytf)
    append_test_log("LSAT_qs_performance_2.csv", answer_df)

    return

# def create_torch_data(data, tokenizer):
#     dataset = data.copy()
#
#     dataset = create_full_q(dataset)
#     dataset = dataset.drop(columns=["question", "prompt", "choice_a", "choice_b", "choice_c", "choice_d", "choice_e"])
#
#
#     rows_list = []
#     for index, row in dataset.iterrows():
#         qnum = torch.tensor(row['q_num'])
#         qtype = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(row['q_type']))
#         fullq = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(row['Full_Q'])))
#         label = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(row['label'])))
#         true_label = torch.cat([x for x in label])
#         q_dict = {"q_num": qnum, "q_type": qtype, "Full_Q": fullq, "label": true_label}
#
#         rows_list.append(q_dict)
#
#     final_dataset = pd.DataFrame(rows_list)
#     return final_dataset


class CustomModel(nn.Module):
    def __init__(self, model_name):

        super(CustomModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.dropout = torch.nn.Dropout(.05)
        self.classifier = nn.Linear(200064, 5)

    def forward(self, input_ids, attn_mask):
        outputs = self.model(input_ids, attention_mask = attn_mask)

        hidden_states = outputs[0]
        outputs = self.dropout(hidden_states)
        outputs = self.classifier(outputs[:, 0, :])

        return outputs





# -------- LOAD DATA -------- #
data_header = ["model", "q_num", "q_type", "response", "correct"]

and_qs = pd.read_csv("AnD_questions.csv")
lsat_qs = pd.read_csv("LSAT_Questions.csv")

lsat_LR = lsat_qs[lsat_qs["q_type"]=="LR"]
lsat_LP = lsat_qs[lsat_qs["q_type"]=="LP"]

lsat_train_LR, lsat_test_LR = train_test_split(lsat_LR, train_size = 50/65, test_size = 15/65, random_state=0)
lsat_train_LP, lsat_test_LP = train_test_split(lsat_LP, train_size = 50/65, test_size = 15/65, random_state=0)
lsat_train_LR, lsat_valid_LR = lsat_train_LR.iloc[:45, ], lsat_train_LR.iloc[45:, ]
lsat_train_LP, lsat_valid_LP = lsat_train_LP.iloc[:45, ], lsat_train_LP.iloc[45:, ]


lsat_train = pd.concat([lsat_train_LR, lsat_train_LP])
lsat_valid = pd.concat([lsat_valid_LR, lsat_valid_LP])
lsat_test = pd.concat([lsat_test_LR, lsat_test_LP])

# lsat_train_dataset = create_torch_data(lsat_train, AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct"))
# print(lsat_train_dataset)


# --------- MODEL: Qwen --------- #
# m_name = "Qwen/QwQ-32B"
# q_model, q_tokenizer = load_model(m_name)
# complete_tests(q_model, q_tokenizer, m_name)
#
#
# q_trained = train_model(q_model)
# complete_tests(q_trained, q_tokenizer, m_name + "_TRAINED")



# # --------- MODEL: MS Phi --------- #
m_name = "microsoft/Phi-4-mini-instruct"
# m_model, m_tokenizer = load_model(m_name)
m_tokenizer = AutoTokenizer.from_pretrained(m_name)

lsat_train_data = create_full_q(lsat_train)
input_ids_tok = m_tokenizer(lsat_train_data['Full_Q'].tolist())
input_ids = torch.tensor(input_ids_tok['input_ids'])
attn_mask = torch.tensor(input_ids_tok['attention_mask'])

# input_tensors = [torch.tensor(x) for x in input_ids['input_ids']]

custom_m = CustomModel(m_name)
print("CUSTOM M:", custom_m)
trained_m = custom_m.forward(input_ids, attn_mask)
print("\nTRAINED M:", trained_m)

# # complete_tests(m_model, m_tokenizer, m_name)
# m_trained = train_model(m_model, m_name)
# complete_tests(trained_m, m_tokenizer, m_name + "_TRAINED")



# # --------- MODEL: Perplexity AI --------- #
# m_name = "perplexity-ai/r1-1776"
# p_model, p_tokenizer = load_model(m_name)
#
# complete_tests(p_model, p_tokenizer, m_name)
# p_trained = train_model(p_model)
# complete_tests(p_model, p_tokenizer, m_name + "_TRAINED")















# # --------- MODEL: Llama --------- #
# m_name = "meta-llama/Llama-3.3-70B-Instruct"
# l_model, l_tokenizer = load_model(m_name)
#
# complete_tests(l_model, l_tokenizer, m_name)
# roberta_trained = train_model(l_model)
# complete_tests(l_model, l_tokenizer, m_name + "_TRAINED")






