# src/lstm_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

from lstm_model import create_lstm_model, NextWordLSTM
from next_token_dataset import setup_data_loaders
from eval_lstm import calculate_rouge_for_completions, generate_examples


class LSTMTrainer:
    """
    Тренер для LSTM модели с полным соответствием требованиям промта
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Устройство: {self.device}")
        
        # Инициализация данных и модели
        self.setup_data_and_model()
        
        # Оптимизатор и функция потерь
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.rouge1_scores = []
        self.rouge2_scores = []
   
#=======================================================  #"""Инициализация данных и модели"""
    """  
    def setup_data_and_model(self):
       
        # Загрузка данных
        (self.train_loader, self.val_loader, self.test_loader, 
         self.token_to_idx, self.idx_to_token) = setup_data_loaders(
            batch_size=self.config['batch_size'],
            sequence_length=self.config['sequence_length']
        )
        
        # Создание модели
        self.model = create_lstm_model(
            vocab_size=len(self.token_to_idx),
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            pad_idx=self.token_to_idx.get('<PAD>', 0)
        ).to(self.device)
        
        print(f"Модель: {sum(p.numel() for p in self.model.parameters()):,} параметров")"""

 #====================================================== """Инициализация данных и модели с ограниченным словарем"""
    def setup_data_and_model(self):
        """Инициализация данных и модели"""
        from next_token_dataset import setup_data_loaders
    
        # Загрузка данных С ОГРАНИЧЕННЫМ СЛОВАРЕМ
        (self.train_loader, self.val_loader, self.test_loader, 
        self.token_to_idx, self.idx_to_token) = setup_data_loaders(
            batch_size=self.config['batch_size'],
            sequence_length=self.config['sequence_length'],
            max_vocab_size=self.config.get('max_vocab_size', 30000)  # ⬅️ ДОБАВЬТЕ ЭТО 
        )
    
        # Создание модели
        self.model = create_lstm_model(
            vocab_size=len(self.token_to_idx),
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            pad_idx=self.token_to_idx.get('<PAD>', 0)
        ).to(self.device)
    
        print(f"Модель: {sum(p.numel() for p in self.model.parameters()):,} параметров")
        print(f"Размер словаря: {len(self.token_to_idx):,} токенов")
    
 # ===================================================== 
    def train_epoch(self, max_batches=None):
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (x_batch, y_batch) in enumerate(progress_bar):
            if max_batches and batch_idx >= max_batches:
                break
                
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            
            # Прямой проход
            self.optimizer.zero_grad()
            logits, _ = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            
            # Обратный проход
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Обновление прогресс-бара
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / batch_count
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_with_rouge(self, max_batches=None):
        """Валидация с вычислением ROUGE метрик"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        batch_count = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(self.val_loader):
                if max_batches and batch_idx >= max_batches:
                    break
                    
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                logits, _ = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item()
                
                # Для accuracy
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                batch_count += 1
                
                # Для ROUGE
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / batch_count
        accuracy = correct / total if total > 0 else 0
        self.val_losses.append(avg_loss)
        
        # Вычисление ROUGE метрик
        from eval_lstm import evaluate_rouge
        rouge1, rouge2 = evaluate_rouge(all_predictions, all_targets, self.idx_to_token)
        self.rouge1_scores.append(rouge1)
        self.rouge2_scores.append(rouge2)
        
        return avg_loss, accuracy, rouge1, rouge2
    
    def train(self, max_batches_per_epoch=None):
        """Основной цикл обучения с ROUGE метриками и примерами"""
        print("Начало обучения LSTM модели...")
        best_rouge1 = 0
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nЭпоха {epoch + 1}/{self.config['num_epochs']}")
            print("-" * 50)
            
            # Обучение
            train_loss = self.train_epoch(max_batches=max_batches_per_epoch)
            
            # Валидация с ROUGE
            val_loss, val_acc, rouge1, rouge2 = self.validate_with_rouge(max_batches=50)
            
            # Вывод метрик
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}") 
            print(f"Val Accuracy: {val_acc:.2%}")
            print(f"ROUGE-1: {rouge1:.4f}")
            print(f"ROUGE-2: {rouge2:.4f}")
            
            # Сохранение лучшей модели
            if rouge1 > best_rouge1:
                best_rouge1 = rouge1
                self.save_model('best_model.pth')
                print(f"🚀 Новая лучшая модель! ROUGE-1: {rouge1:.4f}")
            
            # Показ примеров каждые N эпох
            if (epoch + 1) % self.config.get('show_examples_every', 2) == 0:
                print("\nПримеры генерации:")
                examples = generate_examples(
                    self.model, 
                    self.token_to_idx, 
                    self.idx_to_token,
                    num_examples=2,
                    device=self.device
                )
                for i, (input_text, generated, target) in enumerate(examples):
                    print(f"  {i+1}. Вход: '{input_text}'")
                    print(f"     Сгенерировано: '{generated}'")
            
            # Сохранение чекпоинта
            if (epoch + 1) % self.config.get('save_every', 1) == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Финальное сохранение
        self.save_model('final_model.pth')
        print("\n🎉 Обучение завершено!")
        
        # Итоговые метрики
        print(f"\nЛучшие метрики:")
        print(f"  ROUGE-1: {max(self.rouge1_scores):.4f}")
        print(f"  ROUGE-2: {max(self.rouge2_scores):.4f}")
    
    def save_model(self, filename):
        """Сохранение модели с метриками"""
        models_dir = Path(__file__).parent.parent / 'models'
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'rouge1_scores': self.rouge1_scores,
            'rouge2_scores': self.rouge2_scores
        }, model_path)
        
        print(f"💾 Модель сохранена: {model_path}")


# Оптимальные параметры из промта
def get_optimal_config():
    """Возвращает оптимальную конфигурацию из промта"""
    return {
        'batch_size': 256,
        'sequence_length': 20,
        'embedding_dim': 128,
        'hidden_dim': 256,  # Можно увеличить до 512 если хватает памяти
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 10,
        'save_every': 2,
        'show_examples_every': 2,
        'early_stopping_patience': 3
    }


def main():
    """Основная функция для полного обучения с оптимальными параметрами"""
    config = get_optimal_config()
    
    print("🎯 Обучение с оптимальными параметрами:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    trainer = LSTMTrainer(config)
    trainer.train()  # Полное обучение без ограничений


if __name__ == "__main__":
    main()