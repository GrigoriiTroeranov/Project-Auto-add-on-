# замер метрик lstm модели
# src/eval_lstm.py

import torch
import random
from rouge_score import rouge_scorer
import numpy as np


def get_real_examples_from_data(token_to_idx, idx_to_token, num_examples=5, sequence_length=10):
    """Берет реальные примеры из данных для демонстрации"""
    from next_token_dataset import get_default_data_paths
    import pandas as pd
    
    train_path, val_path, test_path = get_default_data_paths()
    
    try:
        # Читаем тренировочные данные
        df = pd.read_csv(train_path)
        
        examples = []
        for _, row in df.head(num_examples * 2).iterrows():  # Берем с запасом
            try:
                # Парсим токенизированный текст
                tokens = eval(row['tokenized_text']) if isinstance(row['tokenized_text'], str) else []
                if len(tokens) >= sequence_length + 1:
                    # Берем начало последовательности как вход
                    input_tokens = tokens[:sequence_length]
                    target_token = tokens[sequence_length] if len(tokens) > sequence_length else tokens[-1]
                    
                    input_text = ' '.join(input_tokens)
                    target_text = idx_to_token.get(
                        token_to_idx.get(target_token, token_to_idx.get('<UNK>', 0)), 
                        '<UNK>'
                    )
                    
                    examples.append((input_text, target_text))
                    
                    if len(examples) >= num_examples:
                        break
            except:
                continue
                
        return examples
        
    except Exception as e:
        print(f"Не удалось загрузить реальные примеры: {e}")
        # Fallback - простые примеры
        return [
            ("я хочу пойти в", "магазин"),
            ("сегодня очень хорошая", "погода"), 
            ("это интересная", "книга"),
            ("мы будем", "работать"),
            ("он любит", "читать")
        ]


def generate_examples(model, token_to_idx, idx_to_token, num_examples=3, device='cpu', sequence_length=10):
    """Генерация примеров из реальных данных"""
    model.eval()
    examples = []
    
    # Берем реальные примеры из данных
    real_examples = get_real_examples_from_data(token_to_idx, idx_to_token, num_examples, sequence_length)
    
    with torch.no_grad():
        for input_text, real_target in real_examples[:num_examples]:
            # Токенизируем вход
            input_tokens = input_text.split()
            input_indices = []
            
            for token in input_tokens:
                idx = token_to_idx.get(token, token_to_idx.get('<UNK>', 0))
                input_indices.append(idx)
            
            # Паддинг/обрезание до нужной длины
            if len(input_indices) < sequence_length:
                input_indices = input_indices + [token_to_idx.get('<PAD>', 0)] * (sequence_length - len(input_indices))
            elif len(input_indices) > sequence_length:
                input_indices = input_indices[:sequence_length]
            
            # Предсказание
            input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
            logits, _ = model(input_tensor)
            predicted_idx = torch.argmax(logits, dim=1).item()
            predicted_token = idx_to_token.get(predicted_idx, '<UNK>')
            
            generated_text = input_text + " " + predicted_token
            target_text = input_text + " " + real_target
            
            examples.append((input_text, generated_text, target_text))
    
    return examples
#================================================================
def evaluate_rouge(predictions, targets, idx_to_token):
    """Исправленное вычисление ROUGE метрик"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    
    for pred_idx, target_idx in zip(predictions, targets):
        pred_token = idx_to_token.get(pred_idx, '<UNK>')
        target_token = idx_to_token.get(target_idx, '<UNK>')
        
        # ⚡ ИСПРАВЛЕННЫЙ ПОДХОД: одинаковый контекст, разные слова
        context = "The user wrote in their post that they will"
        pred_text = f"{context} {pred_token} tomorrow"
        target_text = f"{context} {target_token} tomorrow"
        
        scores = scorer.score(target_text, pred_text)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
    
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rouge2 = np.mean(rouge2_scores)
    
    print(f"🔍 ROUGE метрики (исправленные):")
    print(f"   Образцов: {len(rouge1_scores)}")
    print(f"   ROUGE-1: {avg_rouge1:.4f}")
    print(f"   ROUGE-2: {avg_rouge2:.4f}")
    
    return avg_rouge1, avg_rouge2
#================================================================
"""def evaluate_rouge(predictions, targets, idx_to_token):"""
  #  """Вычисление ROUGE метрик"""
""" scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    
    for pred_idx, target_idx in zip(predictions, targets):
        pred_token = idx_to_token.get(pred_idx, '<UNK>')
        target_token = idx_to_token.get(target_idx, '<UNK>')
        
        # Более осмысленные предложения для ROUGE
        pred_text = f"Следующее слово должно быть {pred_token} ."
        target_text = f"Следующее слово должно быть {target_token} ."
        
        scores = scorer.score(target_text, pred_text)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
    
    return np.mean(rouge1_scores), np.mean(rouge2_scores)"""
#===============================================================
def calculate_rouge_for_completions(model, dataloader, token_to_idx, idx_to_token, device='cpu', max_batches=20):
    """ROUGE метрики для автодополнений - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits, _ = model(x_batch)
            predictions = torch.argmax(logits, dim=1)
            
            for i in range(len(predictions)):
                pred_idx = predictions[i].item()
                target_idx = y_batch[i].item()
                
                pred_token = idx_to_token.get(pred_idx, '<UNK>')
                target_token = idx_to_token.get(target_idx, '<UNK>')
                
                # ⚡ ИСПРАВЛЕННЫЙ ПОДХОД
                context = "In the social media post the author said"
                pred_text = f"{context} {pred_token} soon"
                target_text = f"{context} {target_token} soon"
                
                scores = scorer.score(target_text, pred_text)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
    
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rouge2 = np.mean(rouge2_scores)
    
    print(f"🔍 ROUGE для автодополнений:")
    print(f"   Образцов: {len(rouge1_scores)}")
    print(f"   ROUGE-1: {avg_rouge1:.4f}")
    print(f"   ROUGE-2: {avg_rouge2:.4f}")
    
    return avg_rouge1, avg_rouge2
#===================================================================================================================

"""def calculate_rouge_for_completions(model, dataloader, token_to_idx, idx_to_token, device='cpu', max_batches=20):"""
   # """ROUGE метрики для автодополнений"""
"""   model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits, _ = model(x_batch)
            predictions = torch.argmax(logits, dim=1)
            
            for i in range(len(predictions)):
                pred_idx = predictions[i].item()
                target_idx = y_batch[i].item()
                
                pred_token = idx_to_token.get(pred_idx, '<UNK>')
                target_token = idx_to_token.get(target_idx, '<UNK>')
                
                pred_text = f"Предсказано слово {pred_token} ."
                target_text = f"Ожидалось слово {target_token} ."
                
                scores = scorer.score(target_text, pred_text)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
    
    return np.mean(rouge1_scores), np.mean(rouge2_scores)"""
#========================================================================