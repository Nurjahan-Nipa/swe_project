import os
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import numpy as np
import spacy
import pandas as pd
from bs4 import BeautifulSoup, Tag, NavigableString
import re
from tqdm import tqdm

PRE_PLACE_HOLDER = "THIS_IS_PLACEHOLDER_FOR_PRE_TAG_ICSE_2023"


def replace_pre_with_placeholder(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    for pre_tag in soup.find_all("pre"):
        if pre_tag.parent:  # 检查 pre_tag 是否有父节点
            pre_tag.insert_before(PRE_PLACE_HOLDER)
            pre_tag.decompose()  # 删除 <pre> 标签及内容
        else:
            print(html_content)
            print("Warning: A <pre> tag has no parent and will be skipped.")

    return str(soup)


def remove_unwanted_tags(html_content):
    allowed_tags = {
        "em",
        "/em",
        "sub",
        "/sub",
        "strong",
        "/strong",
        "code",
        "/code",
        "li",
    }

    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup.find_all(True):
        if tag.name not in allowed_tags:
            tag.unwrap()

    return str(soup)


def split_by_placeholder(input_string):
    segments = re.split(f"({re.escape(PRE_PLACE_HOLDER)})", input_string)
    return [segment for segment in segments if segment.strip()]


def split_string_by_li_tags(html_content):
    results = []
    li_pattern = re.compile(r"<li>(.*?)</li>", re.DOTALL)

    li_matches = li_pattern.finditer(html_content)

    last_end = 0

    for match in li_matches:
        start, end = match.span()
        if last_end < start:
            not_in_li_content = html_content[last_end:start].strip()
            if not_in_li_content:
                results.append([not_in_li_content, "not_in_li"])
        in_li_content = match.group(1).strip()
        if in_li_content:
            results.append([in_li_content, "in_li"])
        last_end = end

    if last_end < len(html_content):
        remaining_content = html_content[last_end:].strip()
        if remaining_content:
            results.append([remaining_content, "not_in_li"])

    true_results = []

    for i in results:
        if i[0].find(PRE_PLACE_HOLDER) != -1:
            pre_chunks = [[j, i[1]] for j in split_by_placeholder(i[0]) if j != ""]
            true_results += pre_chunks
        else:
            true_results += [i]

    pre_indices = []
    before_pre_indices = []
    after_pre_indices = []

    for index, i in enumerate(true_results):
        if i[0] == PRE_PLACE_HOLDER:
            pre_indices.append(index)

    for i in pre_indices:
        if i - 1 >= 0 and i - 1 not in pre_indices:
            before_pre_indices.append(i - 1)
        if i + 1 < len(true_results) and i + 1 not in pre_indices:
            after_pre_indices.append(i + 1)

    to_return = []
    for index, i in enumerate(true_results):
        if index not in pre_indices:
            if index in before_pre_indices and index in after_pre_indices:
                to_return.append([i[0], i[1], "before_and_after_pre"])
            elif index in before_pre_indices:
                to_return.append([i[0], i[1], "before_pre"])
            elif index in after_pre_indices:
                to_return.append([i[0], i[1], "after_pre"])
            else:
                to_return.append([i[0], i[1], "no_pre"])

    return to_return


def sent_tokenize_all_chunk(nlp, chunks):
    def sent_tokenize_a_chunk(nlp, chunk):
        doc = nlp(chunk[0])
        sentences = [sent.text.strip() for sent in doc.sents]
        if chunk[1] == "in_li":
            li_features = [1] * len(sentences)
        else:
            li_features = [0] * len(sentences)
        pre_features = [0] * len(sentences)
        if chunk[2] == "before_pre":
            pre_features[-1] = 1
        if chunk[2] == "after_pre":
            pre_features[0] = 1
        if chunk[2] == "before_and_after_pre":
            pre_features[-1] = 1
            pre_features[0] = 1
        return sentences, li_features, pre_features

    sentences = []
    li_features = []
    pre_features = []
    for c in chunks:
        s, l, p = sent_tokenize_a_chunk(nlp, c)
        sentences += s
        li_features += l
        pre_features += p
    return sentences, li_features, pre_features


def entity_overlap(sentence, tags):
    contain = 0
    for i in tags:
        if sentence.lower().find(i.lower()) != -1:
            contain += 1
    return [float(contain) / len(tags)]


def grammer_check(nlp, sentence):
    result = [0, 0, 0]
    doc = nlp(sentence)
    for token in doc:
        if token.tag_ == "JJR":
            result[0] = 1
        if token.tag_ == "JJS":
            result[1] = 1
    if len(doc) > 0 and doc[0].pos_ == "VERB":
        result[2] = 1
    return result


def bold_text_and_inline_code(sentence):
    result = [0, 0]
    if sentence.find("<strong>") != -1 or sentence.find("</strong>") != -1:
        result[0] = 1
    if sentence.find("<code>") != -1 or sentence.find("</code>") != -1:
        result[1] = 1
    return result


def linguistic_patterns(sentence):
    patterns = [
        "However",
        "First",
        "In short",
        "In this case",
        "In general",
        "Finally",
        "Then",
        "Alternatively",
        "In other words",
        "In addition",
        "In practice",
        "In fact",
        "Otherwise",
        "If you care",
        "In contrast",
        "On the other hand",
        "Below is",
        "Additionally",
        "Furthermore",
    ]

    clean_sentence = re.sub(r"[^\w\s]", "", sentence).lower()

    result = []

    for pattern in patterns:
        clean_pattern = re.sub(r"[^\w\s]", "", pattern).lower()  # 清洗 pattern
        if clean_pattern in clean_sentence:
            result.append(1)
        else:
            result.append(0)

    return result


def bertoverflow(sentence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("jeniya/BERTOverflow")
    model = AutoModel.from_pretrained("jeniya/BERTOverflow").to(device)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(
        device
    )

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state

    attention_mask = inputs["attention_mask"]
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())

    valid_embeddings = last_hidden_state * mask_expanded
    sentence_embedding = valid_embeddings.sum(dim=1) / attention_mask.sum(
        dim=1, keepdim=True
    )

    return sentence_embedding.squeeze(0).cpu().tolist()


def embed_sentences_in_so_post(answer_body, question_tags):
    nlp = spacy.load("en_core_web_sm")
    post = replace_pre_with_placeholder(answer_body)
    post = remove_unwanted_tags(post)
    chunks = split_string_by_li_tags(post)
    sentences, li_features, pre_features = sent_tokenize_all_chunk(nlp, chunks)
    all_sentence_embeddings = []

    for sentence_index, sentence in enumerate(sentences):
        entity_overlap_result = entity_overlap(sentence, question_tags)
        grammer_check_result = grammer_check(nlp, sentence)
        bold_text_and_inline_code_result = bold_text_and_inline_code(sentence)
        linguistic_patterns_result = linguistic_patterns(sentence)
        contain_li = li_features[sentence_index]
        code_adjacent = pre_features[sentence_index]
        if sentence_index == 0:
            position = 1
        else:
            position = 0
        embedding = (
            [code_adjacent, contain_li, position]
            + entity_overlap_result
            + grammer_check_result
            + bold_text_and_inline_code_result
            + linguistic_patterns_result
            + bertoverflow(sentence)
        )
        all_sentence_embeddings.append(embedding)

    return sentences, all_sentence_embeddings

question_title = "This should be replaced with real question title."
answer_body = "This should be replaced with SO post with html tags."
question_tags = ["Java", "Python"]

def load_model(model_path):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    print(f"Model loaded from {model_path}.")
    return model


def predict_probabilities(input_vector, model):
    input_vector = input_vector.reshape(1, -1)
    return model.predict_proba(input_vector)[0]


type_model_path = "../models/question_classifier.pkl"
type_classifier = load_model(type_model_path)

type_classifiers = {
    "1": load_model("../models/1.pkl"),
    "2": load_model("../models/2.pkl"),
    "3": load_model("../models/3.pkl"),
}

question_embedding = np.array(bertoverflow(question_title))
sentences, sentence_embeddings = embed_sentences_in_so_post(answer_body, question_tags)

for sentence, sentence_embedding in zip(sentences, sentence_embeddings):
    type_probabilities = predict_probabilities(question_embedding, type_classifier)
    type_specific_predictions = {}
    for type_id, classifier in type_classifiers.items():
        type_specific_predictions[type_id] = predict_probabilities(
            np.array(sentence_embedding), classifier
        )

    final_probabilities = np.zeros_like(list(type_specific_predictions.values())[0])
    for type_id, type_proba in type_specific_predictions.items():
        weight = type_probabilities[int(type_id) - 1]
        final_probabilities += weight * type_proba

    print("For sentence:", sentence)
    print(f"The possibility of it being important is: {final_probabilities[1]:.4f}")


