# код с запуском и замером качества трансформера
# src/eval_transformer_pipeline.py

"""
Модуль для оценки предобученной модели трансформера distilgpt2
Генерация текстов, валидация модели, замер ROUGE метрик
"""

from transformers import pipeline
from rouge_score import rouge_scorer
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Импортируем функции из существующих модулей
try:
    from eval_lstm import evaluate_rouge, show_examples
except ImportError:
    # Fallback реализации если модуль не найден
    def evaluate_rouge(predictions, targets):
        """Вычисление ROUGE метрик"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        rouge1_scores = [scorer.score(t, p)['rouge1'].fmeasure for p, t in zip(predictions, targets)]
        rouge2_scores = [scorer.score(t, p)['rouge2'].fmeasure for p, t in zip(predictions, targets)]
        return sum(rouge1_scores)/len(rouge1_scores), sum(rouge2_scores)/len(rouge2_scores)
    
    def show_examples(predictions, num_examples=3):
        """Показ примеров предсказаний"""
        for i, (prompt, generated, target) in enumerate(predictions[:num_examples]):
            print(f"Пример {i+1}:")
            print(f"Вход: '{prompt}'")
            print(f"Сгенерировано: '{generated}'")
            print(f"Ожидалось: '{target}'\n")


class TransformerEvaluator:
    """Оценщик для трансформерной модели distilgpt2"""
    
    def __init__(self, model_name="distilgpt2"):
        self.generator = pipeline("text-generation", model=model_name)
        print(f"✅ Модель {model_name} загружена")
    
    def load_data(self, split="val"):
        """Загрузка данных из папки data/"""
        data_path = Path(__file__).parent.parent / 'data' / f'{split}.csv'
        df = pd.read_csv(data_path)
        
        # Используем cleaned_text или text
        text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
        texts = [str(t) for t in df[text_col] if isinstance(t, str) and len(t.split()) >= 8]
        
        print(f"📊 Загружено {len(texts)} текстов")
        return texts
    
    def prepare_pairs(self, texts, num_samples=100):
        """Подготовка пар (вход, цель) - предсказываем последнюю четверть"""
        pairs = []
        for text in texts[:num_samples]:
            words = text.split()
            if len(words) >= 10:
                split_point = int(len(words) * 0.75)  # Берем 75% как вход
                input_text = ' '.join(words[:split_point])
                target_text = ' '.join(words[split_point:])
                pairs.append((input_text, target_text))
        return pairs
    
    def generate_text(self, prompt, **kwargs):
        """Генерация продолжения текста"""
        result = self.generator(prompt, **kwargs)[0]["generated_text"]
        return result[len(prompt):].strip() if result.startswith(prompt) else result
    
    def evaluate(self, text_pairs, **gen_kwargs):
        """Полная оценка модели с ROUGE метриками"""
        predictions = []
        targets = []
        
        for prompt, target in tqdm(text_pairs, desc="Генерация"):
            generated = self.generate_text(prompt, **gen_kwargs)
            if generated:
                predictions.append(generated)
                targets.append(target)
        
        rouge1, rouge2 = evaluate_rouge(predictions, targets)
        
        # Собираем примеры для показа
        examples = [(p, g, t) for (p, t), g in zip(text_pairs, predictions)]
        
        return {
            'rouge1': rouge1,
            'rouge2': rouge2,
            'examples': examples,
            'num_samples': len(predictions)
        }
    
    def tune_parameters(self, text_pairs, param_grid):
        """Подбор лучших параметров генерации"""
        best_score = 0
        best_params = None
        
        for params in param_grid:
            results = self.evaluate(text_pairs[:20], **params)  # Быстрая оценка
            if results['rouge1'] > best_score:
                best_score = results['rouge1']
                best_params = params
        
        return best_params, best_score


def main():
    """Основная функция оценки"""
    print("🚀 Оценка трансформерной модели distilgpt2")
    
    # Инициализация
    evaluator = TransformerEvaluator()
    
    # Загрузка и подготовка данных
    texts = evaluator.load_data("val")
    text_pairs = evaluator.prepare_pairs(texts, 100)
    
    # Сетка параметров для подбора
    param_grid = [
        {'max_length': 30, 'do_sample': True, 'top_k': 50, 'temperature': 0.7},
        {'max_length': 40, 'do_sample': True, 'top_k': 50, 'temperature': 0.8},
        {'max_length': 50, 'do_sample': True, 'top_k': 50, 'temperature': 0.9},
        {'max_length': 30, 'do_sample': False},  # greedy decoding
    ]
    
    # Подбор параметров
    best_params, best_score = evaluator.tune_parameters(text_pairs, param_grid)
    print(f"🎯 Лучшие параметры: {best_params}")
    print(f"ROUGE-1: {best_score:.4f}")
    
    # Финальная оценка
    print("\n📊 Финальная оценка...")
    results = evaluator.evaluate(text_pairs, **best_params)
    
    # Результаты
    print(f"\n📈 Результаты на {results['num_samples']} samples:")
    print(f"ROUGE-1: {results['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge2']:.4f}")
    
    # Примеры
    print("\n📝 Примеры предсказаний:")
    show_examples(results['examples'], 3)
    
    # Демонстрация как в промте
    print("\n🎲 Демонстрация генерации:")
    test_prompt = "I think that artificial intelligence"
    result = evaluator.generator(test_prompt, max_length=20, do_sample=True, top_k=50)
    print(f"Вход: '{test_prompt}'")
    print(f"Результат: '{result[0]['generated_text']}'")


if __name__ == "__main__":
    main()