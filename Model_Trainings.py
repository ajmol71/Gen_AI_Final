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

import time


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
    for index, row in test_data.iterrows():
        question, prompt = row["question"], row["prompt"]

        full_question = f" Here are the options: A. {row["choice_a"]}  B. {row["choice_b"]}, C. {row["choice_c"]}, D. {row["choice_d"]}, E. {row["choice_e"]}" + question

        time_start = time.time()

        inputs = tokenizer(full_question, prompt, return_tensors=pytf)
        outputs = model(**inputs)

        if pytf == "tf":
            answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
            answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
            predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
            answer = tokenizer.decode(predict_answer_tokens)
        elif pytf == "pt":
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

        if row[row['label']] in answer:
            correct = 1
        else:
            correct = 0

        time_now = time.time()
        time_took = time_now - time_start

        print("\nQUESTION: ", row['question'])
        print("ANSWER: ", answer)
        print("TIME TOOK:", time_took)

        q_dict = {"model": m_name, "q_num": row["q_num"], "q_type": row["q_type"], "answer": answer, "correct": correct, "time_took": time_took}

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

def append_test_log(file_name, new_data):
    with open(file_name, 'a') as file:
        writer = csv.writer(file)

        for index, row in new_data.iterrows():
            writer.writerow(row)
    return

def complete_tests(model, tokenizer, m_name, pytf="pt"):
    # AnD Tests
    answer_df = test_model(tokenizer, model, m_name, and_qs, pytf)
    append_test_log("AnD_qs_performance_final.csv", answer_df)

    # LSAT Tests
    answer_df = test_model(tokenizer, model, m_name, lsat_test, pytf)
    append_test_log("LSAT_qs_performance_final.csv", answer_df)

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
    def __init__(self, model, m_name):

        super(CustomModel, self).__init__()
        self.model = model

        self.dropout = torch.nn.Dropout(.25)
        if m_name == "roberta":
            self.classifier = nn.Linear(193, 5)
        elif m_name == "distilbert":
            self.classifier = nn.Linear(185, 5)
        elif m_name == "qwen":
            self.classifier = nn.Linear(188, 5)
        elif m_name == "T5":
            self.classifier = nn.Linear(204, 5)
        else:
            self.classifier = nn.Linear(204, 5)
        # roberta: 193; distilbert: 185; qwen 188; t5 204

    def forward(self, input_ids, attn_mask):
        outputs = self.model(input_ids, attention_mask = attn_mask)

        hidden_states = outputs[0]

        self.dropout(hidden_states)
        self.classifier(outputs['end_logits'])

        return self.model
        # return outputs


# def create_data_dict(dataset):
#     dicts = []
#     for index, row in dataset.iterrows():
#         answer = row['label']
#         context_text = row['prompt'] + f" Here are the options: A. {row["choice_a"]}  B. {row["choice_b"]}, C. {row["choice_c"]}, D. {row["choice_d"]}, E. {row["choice_e"]}"
#         indices = context_text.split(' ')
#         answer_toks = answer.split(' ')
#         ans_index = indices.iloc[answer_toks[0]]
#         q_dict = {'answers': {'answer_start': [ans_index],
#                 'text': ['answer']}, 'context': context_text,'id': row['q_num'],
#                   'question': row['question']}
#         dicts.append(q_dict)



# -------- LOAD DATA -------- #
data_header = ["model", "q_num", "q_type", "response", "correct", "time_took"]

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

lsat_train_data = create_full_q(lsat_train)
lsat_valid_data = create_full_q(lsat_valid)


# lsat_train_dataset = create_torch_data(lsat_train, AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct"))
# print(lsat_train_dataset)


# --------- MODEL: RoBERTa --------- #
# from transformers import AutoTokenizer, RobertaForQuestionAnswering
# m_name = "roberta"
# r_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
# r_model = RobertaForQuestionAnswering.from_pretrained("FacebookAI/roberta-base")
#
# input_ids_tok = r_tokenizer(lsat_train_data['Full_Q'].tolist())
# input_ids = torch.tensor(input_ids_tok['input_ids'])
# attn_mask = torch.tensor(input_ids_tok['attention_mask'])
# valid_ids_tok = r_tokenizer(lsat_valid_data['Full_Q'].tolist())
#
#
# complete_tests(r_model, r_tokenizer, m_name)
#
# custom_r = CustomModel(r_model, m_name)
# # print("\n\nCUSTOM M:", custom_d)
# trained_r = custom_r.forward(input_ids, attn_mask)
# # print("\nTRAINED M:", trained_d)
#
# complete_tests(trained_r, r_tokenizer, m_name + "_TRAINED")


# # --------- MODEL: DistilBert --------- #
# from transformers import AutoTokenizer, DistilBertForQuestionAnswering
# m_name = "distilbert"
# d_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# d_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
#
# input_ids_tok = d_tokenizer(lsat_train_data['Full_Q'].tolist())
# input_ids = torch.tensor(input_ids_tok['input_ids'])
# attn_mask = torch.tensor(input_ids_tok['attention_mask'])
# valid_ids_tok = d_tokenizer(lsat_valid_data['Full_Q'].tolist())
#
#
# complete_tests(d_model, d_tokenizer, m_name)
#
# custom_d = CustomModel(d_model, m_name)
# # print("\n\nCUSTOM M:", custom_d)
# trained_d = custom_d.forward(input_ids, attn_mask)
# # print("\nTRAINED M:", trained_d)
#
# complete_tests(trained_d, d_tokenizer, m_name + "_TRAINED")



# # --------- MODEL: Qwen --------- #
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
m_name = "qwen"
q_model = AutoModelForQuestionAnswering.from_pretrained("knowledgator/Qwen-encoder-0.5B")
q_tokenizer = AutoTokenizer.from_pretrained("knowledgator/Qwen-encoder-0.5B")

input_ids_tok = q_tokenizer(lsat_train_data['Full_Q'].tolist())
input_ids = torch.tensor(input_ids_tok['input_ids'])
attn_mask = torch.tensor(input_ids_tok['attention_mask'])
valid_ids_tok = q_tokenizer(lsat_valid_data['Full_Q'].tolist())

complete_tests(q_model, q_tokenizer, m_name)

custom_q = CustomModel(q_model, m_name)
# print("\n\nCUSTOM M:", custom_d)
trained_q = custom_q.forward(input_ids, attn_mask)
# print("\nTRAINED M:", trained_d)

complete_tests(trained_q, q_tokenizer, m_name + "_TRAINED")



# # --------- MODEL: T5 --------- #
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
m_name = "T5"
t_model = AutoModelForQuestionAnswering.from_pretrained("mrm8488/t5-base-finetuned-quartz")
t_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-quartz")

input_ids_tok = t_tokenizer(lsat_train_data['Full_Q'].tolist())
input_ids = torch.tensor(input_ids_tok['input_ids'])
attn_mask = torch.tensor(input_ids_tok['attention_mask'])
valid_ids_tok = t_tokenizer(lsat_valid_data['Full_Q'].tolist())

complete_tests(t_model, t_tokenizer, m_name)

custom_t = CustomModel(t_model, m_name)
# print("\n\nCUSTOM M:", custom_d)
trained_t = custom_t.forward(input_ids, attn_mask)
# print("\nTRAINED M:", trained_d)

complete_tests(trained_t, t_tokenizer, m_name + "_TRAINED")










# # # --------- MODEL: Llama encoder --------- #
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# m_name = "llama"
# l_model = AutoModelForQuestionAnswering.from_pretrained("knowledgator/Llama-encoder-1.0B")
# l_tokenizer = AutoTokenizer.from_pretrained("knowledgator/Llama-encoder-1.0B")
#
# input_ids_tok = l_tokenizer(lsat_train_data['Full_Q'].tolist())
# input_ids = torch.tensor(input_ids_tok['input_ids'])
# attn_mask = torch.tensor(input_ids_tok['attention_mask'])
# valid_ids_tok = l_tokenizer(lsat_valid_data['Full_Q'].tolist())
#
# complete_tests(l_model, l_tokenizer, m_name)
#
# custom_l = CustomModel(l_model)
# # print("\n\nCUSTOM M:", custom_d)
# trained_l = custom_l.forward(input_ids, attn_mask)
# # print("\nTRAINED M:", trained_d)
#
# complete_tests(trained_l, l_tokenizer, m_name + "_TRAINED")









# # # --------- MODEL: RWKV7 G1 --------- #
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# m_name = "rwkv7_G1"
# g_model = AutoModelForQuestionAnswering.from_pretrained("fla-hub/rwkv7-0.1B-g1", trust_remote_code=True)
# g_tokenizer = AutoTokenizer.from_pretrained("fla-hub/rwkv7-0.1B-g1", trust_remote_code=True)
#
# input_ids_tok = g_tokenizer(lsat_train_data['Full_Q'].tolist())
# input_ids = torch.tensor(input_ids_tok['input_ids'])
# attn_mask = torch.tensor(input_ids_tok['attention_mask'])
# valid_ids_tok = g_tokenizer(lsat_valid_data['Full_Q'].tolist())
#
# complete_tests(g_model, g_tokenizer, m_name)
#
# custom_g = CustomModel(g_model)
# # print("\n\nCUSTOM M:", custom_d)
# trained_g = custom_g.forward(input_ids, attn_mask)
# # print("\nTRAINED M:", trained_d)
#
# complete_tests(trained_g, g_tokenizer, m_name + "_TRAINED")







