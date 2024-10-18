import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import torch.nn as nn
from training_data import TRAINING_DATA
import re
from nltk.translate.bleu_score import sentence_bleu

class PromptToQueryGenerator:
    def __init__(self, model_name: str = "t5-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def generate_query(self, prompt: str) -> str:
        input_text = f"convert to mongodb query: {prompt}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(input_ids, max_length=512, num_return_sequences=1, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def prepare_training_data(self, data: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_texts = [f"convert to mongodb query: {prompt}" for prompt, _ in data]
        target_texts = [query for _, query in data]

        input_encodings = self.tokenizer(input_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        target_encodings = self.tokenizer(target_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

        return input_encodings.input_ids, target_encodings.input_ids

    def train(self, train_data: List[Tuple[str, str]], val_data: List[Tuple[str, str]], num_epochs: int = 50, batch_size: int = 32, patience: int = 5, accumulation_steps: int = 2):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        input_ids, target_ids = self.prepare_training_data(train_data)
        dataset = torch.utils.data.TensorDataset(input_ids, target_ids)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                input_ids, target_ids = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids=input_ids, labels=target_ids)
                loss = outputs.loss / accumulation_steps
                total_loss += loss.item() * accumulation_steps

                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            scheduler.step()

            # Validation
            val_loss, bleu_score = self.evaluate(val_data, batch_size)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}, BLEU Score: {bleu_score:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("best_model")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            self.model.train()

    def evaluate(self, val_data: List[Tuple[str, str]], batch_size: int):
        self.model.eval()
        val_loss = 0
        bleu_scores = []
        with torch.no_grad():
            val_input_ids, val_target_ids = self.prepare_training_data(val_data)
            val_dataset = torch.utils.data.TensorDataset(val_input_ids, val_target_ids)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
            
            for batch in val_dataloader:
                input_ids, target_ids = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids=input_ids, labels=target_ids)
                val_loss += outputs.loss.item()

                # Calculate BLEU score
                generated = self.model.generate(input_ids)
                for gen, target in zip(generated, target_ids):
                    gen_text = self.tokenizer.decode(gen, skip_special_tokens=True)
                    target_text = self.tokenizer.decode(target, skip_special_tokens=True)
                    bleu_scores.append(sentence_bleu([target_text.split()], gen_text.split()))

        return val_loss / len(val_dataloader), sum(bleu_scores) / len(bleu_scores)

    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load_model(cls, path: str):
        generator = cls(model_name=path)
        return generator

# Example usage
if __name__ == "__main__":
    generator = PromptToQueryGenerator(model_name="t5-large")

    prompt = "Find all users who are over 30 years old and live in New York"
    query = generator.generate_query(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated Query before training: {query}")

    train_data, val_data = train_test_split(TRAINING_DATA, test_size=0.1, random_state=42)

    generator.train(train_data, val_data, num_epochs=50, batch_size=32, patience=5)

    query = generator.generate_query(prompt)
    print(f"Generated Query after training: {query}")

    generator.save_model("prompt_to_query_model")

    loaded_generator = PromptToQueryGenerator.load_model("prompt_to_query_model")
    query = loaded_generator.generate_query(prompt)
    print(f"Generated Query from loaded model: {query}")
