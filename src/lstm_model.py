# src/lstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import random


class NextWordLSTM(nn.Module):
    """
    LSTM модель для предсказания следующего токена.
    Поддерживает два режима:
    - Обучение: предсказание одного следующего токена
    - Генерация: генерация последовательности токенов
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        """
        Args:
            vocab_size: размер словаря
            embedding_dim: размерность эмбеддингов
            hidden_dim: размерность скрытого состояния LSTM
            num_layers: количество слоев LSTM
            dropout: вероятность dropout
            pad_idx: индекс паддинг-токена
        """
        super(NextWordLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        
        # Эмбеддинг слой
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=pad_idx
        )
        
        # LSTM слои
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Выходной слой
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов модели"""
        # Инициализация эмбеддингов
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        if self.pad_idx is not None:
            nn.init.constant_(self.embedding.weight[self.pad_idx], 0)
        
        # Инициализация LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Инициализация forget gate bias для лучшего обучения
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1.0)
        
        # Инициализация выходного слоя
        nn.init.normal_(self.fc.weight, mean=0, std=0.1)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Прямой проход для обучения.
        
        Args:
            x: входные токены [batch_size, seq_len]
            hidden: скрытое состояние LSTM
            
        Returns:
            logits: выходные логиты [batch_size, vocab_size]
            hidden: обновленное скрытое состояние
        """
        # Эмбеддинг
        emb = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        emb = self.dropout(emb)
        
        # LSTM
        lstm_out, hidden = self.lstm(emb, hidden)  # [batch_size, seq_len, hidden_dim]
        lstm_out = self.dropout(lstm_out)
        
        # Берем последний скрытый状态 для предсказания следующего токена
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Выходной слой
        logits = self.fc(last_hidden)  # [batch_size, vocab_size]
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Инициализация скрытого состояния.
        
        Args:
            batch_size: размер батча
            device: устройство (cpu/gpu)
            
        Returns:
            Инициализированное скрытое состояние
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def generate(
        self,
        input_sequence: List[str],
        token_to_idx: dict,
        idx_to_token: dict,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = True
    ) -> str:
        """
        Генерация текста на основе начальной последовательности.
        
        Args:
            input_sequence: начальная последовательность токенов
            token_to_idx: словарь токен->индекс
            idx_to_token: словарь индекс->токен
            max_length: максимальная длина генерируемой последовательности
            temperature: температура для sampling
            top_k: использовать только top-k токенов
            do_sample: использовать ли sampling
            
        Returns:
            Сгенерированная последовательность текста
        """
        self.eval()
        
        # Преобразуем входные токены в индексы
        input_indices = []
        for token in input_sequence:
            idx = token_to_idx.get(token, token_to_idx.get('<UNK>', 0))
            input_indices.append(idx)
        
        generated_tokens = input_sequence.copy()
        current_sequence = torch.tensor([input_indices], dtype=torch.long)
        
        # Инициализируем скрытое состояние
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Прямой проход
                logits, hidden = self.forward(current_sequence, hidden)
                
                # Применяем temperature
                logits = logits / temperature
                
                # Получаем вероятности
                probs = F.softmax(logits, dim=-1)
                
                # Применяем top-k filtering если нужно
                if top_k is not None:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                    probs = torch.zeros_like(probs)
                    probs.scatter_(1, top_k_indices, top_k_probs)
                
                # Выбираем следующий токен
                if do_sample:
                    next_token_idx = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token_idx = torch.argmax(probs, dim=-1).item()
                
                # Преобразуем индекс в токен
                next_token = idx_to_token.get(next_token_idx, '<UNK>')
                
                # Добавляем к сгенерированной последовательности
                generated_tokens.append(next_token)
                
                # Обновляем текущую последовательность для следующего шага
                current_sequence = torch.tensor([[next_token_idx]], dtype=torch.long)
                
                # Останавливаемся если достигли конца последовательности
                if next_token in ['<EOS>', '.', '!', '?']:
                    break
        
        return ' '.join(generated_tokens)
    
    def generate_from_text(
        self,
        text: str,
        token_to_idx: dict,
        idx_to_token: dict,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = True
    ) -> str:
        """
        Генерация текста из строки (простой интерфейс).
        
        Args:
            text: начальный текст
            token_to_idx: словарь токен->индекс
            idx_to_token: словарь индекс->токен
            max_length: максимальная длина генерируемой последовательности
            temperature: температура для sampling
            top_k: использовать только top-k токенов
            do_sample: использовать ли sampling
            
        Returns:
            Сгенерированный текст
        """
        # Простая токенизация (можно заменить на более сложную)
        input_sequence = text.lower().split()
        return self.generate(
            input_sequence,
            token_to_idx,
            idx_to_token,
            max_length,
            temperature,
            top_k,
            do_sample
        )


def count_parameters(model: nn.Module) -> int:
    """
    Подсчет количества обучаемых параметров модели.
    
    Args:
        model: модель PyTorch
        
    Returns:
        Количество параметров
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_lstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    pad_idx: int = 0
) -> NextWordLSTM:
    """
    Фабричная функция для создания LSTM модели.
    
    Args:
        vocab_size: размер словаря
        embedding_dim: размерность эмбеддингов
        hidden_dim: размерность скрытого состояния
        num_layers: количество слоев LSTM
        dropout: вероятность dropout
        pad_idx: индекс паддинг-токена
        
    Returns:
        Созданная LSTM модель
    """
    model = NextWordLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=pad_idx
    )
    
    print(f"Создана LSTM модель с {count_parameters(model):,} параметрами")
    print(f"Архитектура: embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, "
          f"num_layers={num_layers}, dropout={dropout}")
    
    return model


# Пример использования модели
if __name__ == "__main__":
    # Тестируем модель
    vocab_size = 10000
    model = create_lstm_model(vocab_size)
    
    # Тестовый вход
    batch_size, seq_len = 4, 10
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Прямой проход
    logits, hidden = model(test_input)
    print(f"Вход: {test_input.shape}")
    print(f"Выход: {logits.shape}")
    print(f"Скрытое состояние: {hidden[0].shape}")
    
    # Тест генерации (с mock словарями)
    mock_token_to_idx = {'я': 1, 'собираюсь': 2, 'купить': 3, 'продукты': 4, '.': 5}
    mock_idx_to_token = {v: k for k, v in mock_token_to_idx.items()}
    
    try:
        generated = model.generate(
            input_sequence=['я', 'собираюсь', 'купить'],
            token_to_idx=mock_token_to_idx,
            idx_to_token=mock_idx_to_token,
            max_length=10,
            temperature=0.8,
            top_k=50,
            do_sample=True
        )
        print(f"Сгенерированный текст: {generated}")
    except Exception as e:
        print(f"Генерация не удалась (ожидаемо для теста): {e}")