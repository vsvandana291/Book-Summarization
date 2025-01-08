from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from accelerate import Accelerator, DataLoaderConfiguration

def get_accelerator_config():
    return DataLoaderConfiguration(dispatch_batches=None, split_batches=False)

def prepare_dataset(paragraphs):
    inputs = [f"summarize: {paragraph}" for paragraph in paragraphs if len(paragraph.strip()) > 0]
    targets = paragraphs  
    print(f"Number of samples in the dataset: {len(inputs)}")
    dataset = Dataset.from_dict({"input_text": inputs, "target_text": targets})
    return dataset

def train_chapter(dataset, model_name='t5-small'):
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    def preprocess_function(examples):
        inputs = examples['input_text']
        targets = examples['target_text']
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        labels = tokenizer(targets, max_length=512, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(dispatch_batches=None, split_batches=False))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,  
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(f"./{model_name}-model")
    tokenizer.save_pretrained(f"./{model_name}-tokenizer")
    print(f"Model and tokenizer saved to ./{model_name}-model and ./{model_name}-tokenizer")

    return model, tokenizer

def load_model_and_tokenizer(model_name):
    print("Load saved model and tokenizer")
    if model_name == 't5-small':
        model = T5ForConditionalGeneration.from_pretrained("./t5-small-model")
        tokenizer = T5Tokenizer.from_pretrained("./t5-small-tokenizer")
    else:
        raise ValueError("Invalid model name")
    print("Loaded saved model and tokenizer")
    return model, tokenizer



