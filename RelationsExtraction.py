# -*- coding: utf-8 -*-

import json
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments, BertTokenizerFast
from datasets import Dataset

# 全局参数
label_list = ['O', 'B-SUBJ', 'I-SUBJ', 'B-OBJ', 'I-OBJ', 'B-REL', 'I-REL']  # 更新标签列表
num_labels = len(label_list)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")


# 数据准备
def prepare_data(train_data):
    dataset = Dataset.from_list(train_data)
    return dataset


# 加载分词器
def load_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    return tokenizer


# 数据预处理（分词与标签对齐）
def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    all_labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]] if label[word_idx].startswith('I') else -100)
            previous_word_idx = word_idx

        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


# 定义模型
def load_model(num_labels):
    model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)
    model.to(device)  # 将模型移到相应的设备
    return model


# 保存模型和分词器
def save_model(model, tokenizer, output_dir="./gpu_saved_model"):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_model_and_tokenizer(output_dir="./gpu_saved_model"):
    tokenizer = BertTokenizerFast.from_pretrained(output_dir)
    model = BertForTokenClassification.from_pretrained(output_dir)
    model.to(device)  # 将模型移到相应的设备
    # 启用fp16模式
    if torch.cuda.is_available():
        model.half()  # 切换到fp16模式
    return tokenizer, model


# 训练模型
def train_model(dataset, tokenizer, model):
    dataset = dataset.train_test_split(train_size=0.7)  # 调整以更合理地分配训练和验证集
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    tokenized_train_dataset = train_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    tokenized_eval_dataset = eval_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir='../gpu_results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        fp16=True,
        logging_dir='./gpu_logs',
        logging_steps=10,
        report_to="tensorboard"  # 启用 TensorBoard
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer
    )

    print("Starting training...")
    trainer.train()

    return trainer.state.log_history


def plot_losses(losses, output_file):
    # 提取包含训练损失的条目
    train_losses = [x['loss'] for x in losses if 'loss' in x and 'epoch' in x]

    # 确认训练损失与 epoch 数量匹配
    steps = range(1, len(train_losses) + 1)

    plt.plot(steps, train_losses, label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_file)


# 定义提取实体和关系的函数
def extract_entities_and_relations(text, tokenizer, model, id2label):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    inputs = {key: value.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for key, value in
              inputs.items()}  # 确保输入移至 GPU
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    labels = [id2label[id] for id in predictions[0].tolist()]

    # 打印调试信息，帮助排查问题
    # print("Tokenized input IDs:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
    # print("Predictions:", predictions)
    # print("Tokens:", tokens)
    # print("Labels:", labels)

    triples = []
    current_subject = ""
    current_object = ""
    current_relation = ""
    state = None  # None, "SUBJECT", "OBJECT", "RELATION"

    for token, label in zip(tokens[1:], labels[1:]):  # 跳过 [CLS]
        if token in ["[SEP]", "[CLS]", "[PAD]"]:
            continue

        # print(
        #     f"Token: {token}, Label: {label}, State: {state}, Subject: {current_subject}, Relation: {current_relation}, Object: {current_object}")

        if label == "O":
            if current_subject and current_relation and current_object:
                triples.append((current_subject, current_relation, current_object))
                current_subject = ""
                current_relation = ""
                current_object = ""
            state = None
            continue

        if label.startswith("B-SUBJ"):
            if current_subject and current_relation and current_object:
                triples.append((current_subject, current_relation, current_object))
            current_subject = token.replace("##", "")
            current_relation = ""
            current_object = ""
            state = "SUBJECT"

        elif label.startswith("I-SUBJ") and state == "SUBJECT":
            current_subject += token.replace("##", "")

        elif label.startswith("B-OBJ"):
            if current_subject and current_relation and current_object:
                triples.append((current_subject, current_relation, current_object))
            current_object = token.replace("##", "")
            current_relation = ""
            state = "OBJECT"

        elif label.startswith("I-OBJ") and state == "OBJECT":
            current_object += token.replace("##", "")

        elif label.startswith("B-REL"):
            if current_relation:
                if current_subject and current_relation and current_object:
                    triples.append((current_subject, current_relation, current_object))
            current_relation = token.replace("##", "")
            state = "RELATION"

        elif label.startswith("I-REL") and state == "RELATION":
            current_relation += token.replace("##", "")

    # 最后一对 SUBJECT + RELATION + OBJECT 的处理
    if current_subject and current_relation and current_object:
        triples.append((current_subject, current_relation, current_object))

    return triples


# 训练并保存模型
def train_and_save_model():
    with open('../datajson/SO_test_gened_bio_data.json', encoding='utf-8') as f:
        train_data = json.load(f)
    dataset = prepare_data(train_data)
    tokenizer = load_tokenizer()
    model = load_model(num_labels)

    # 训练模型并获取日志
    log_history = train_model(dataset, tokenizer, model)

    # 保存模型和分词器
    save_model(model, tokenizer)

    # 绘制损失值图像
    plot_losses(log_history, '../datajpg/train_loss.png')


# 使用已经训练好的模型进行测试和调试
def test_model():
    tokenizer, model = load_model_and_tokenizer()  # 请替换为您的模型路径
    test_text = ("从正门进来，观云楼在正门的南边，看见观云楼后，朝观云楼那边走，到时你可以看见丹桂苑在观云楼的东南角。"
                 "看见丹桂苑后，一直走，分别是不同的研究生公寓，研究生公寓在丹桂苑的正东。"
                 "报到处在篮球场，位于研究生公寓的北面。"
                 "祁连堂则靠近研究生公寓的东边，体育馆也在祁连堂的东边，兰大出版社则在体育馆南边，校医院又在出版社东边。"
                 "研究生公寓1#也在校医院的东边，凌云楼则在研究生公寓1#的东边。"
                 "看见凌云楼后，就能找到思雨楼了，因为思雨楼在它那个就是凌云楼的正南方。"
                 "这时，可以看到南二门在思雨楼的西边。同时，也可以注意到南二门在研究生公寓2#的东边。")
    triples = extract_entities_and_relations(test_text, tokenizer, model, id2label)
    print("Extracted triples:", triples)


if __name__ == '__main__':
    # train_and_save_model()
    test_model()
