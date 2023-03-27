#!/usr/bin/env python3
import json


def parse_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    qa_pairs = []
    for document in data["data"]:
        for paragraph in document["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]

                answers = {
                    "text": [],
                    "answer_start": [],
                }

                for answer in qa["answers"]:
                    answer_text = answer["text"]
                    answer_start = answer["answer_start"]

                    answers["text"].append(answer_text)
                    answers["answer_start"].append(answer_start)

                qa_pairs.append(
                    {
                        "id": qa["id"],
                        "title": document["title"],
                        "context": context,
                        "question": question,
                        "answers": answers,
                    }
                )

    return qa_pairs


def save_data(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    data = parse_data("data/spoken_train-v1.1.json")
    save_data(data, "data/train.json")

    data = parse_data("data/spoken_test-v1.1.json")
    save_data(data, "data/validation.json")
