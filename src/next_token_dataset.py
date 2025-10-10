# src/next_token_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import ast


class NextTokenDataset(Dataset):
    """
    Dataset для предсказания следующего токена.
    Формирует пары (X, Y) где:
    - X: последовательность токенов
    - Y: следующий токен после последовательности X
    """
    
    def __init__(self, data_path: str, sequence_length: int = 10):
        """
        Args:
            data_path: путь к CSV файлу с данными
            sequence_length: длина входной последовательности
        """
        self.sequence_length = sequence_length
        self.samples = []
        
        # Загружаем данные
        df = pd.read_csv(data_path)
        
        # Обрабатываем каждую строку
        for _, row in df.iterrows():
            tokenized_text = self._parse_tokenized_text(row['tokenized_text'])
            
            # Создаем обучающие примеры для этой последовательности
            if len(tokenized_text) > sequence_length:
                self._create_samples_from_sequence(tokenized_text)
    
    def _parse_tokenized_text(self, tokenized_text_str: str) -> List[str]:
        """Парсит строку с токенизированным текстом в список токенов"""
        try:
            # Пытаемся распарсить как список Python
            return ast.literal_eval(tokenized_text_str)
        except:
            # Если не получается, разбиваем по запятым и убираем кавычки
            tokens = str(tokenized_text_str).strip("[]").split("', '")
            tokens = [token.strip("'\"") for token in tokens if token.strip("'\"")]
            return tokens
    
    def _create_samples_from_sequence(self, tokens: List[str]):
        """Создает обучающие примеры из последовательности токенов"""
        # Для каждой позиции создаем пример (все_токены_до -> следующий_токен)
        for i in range(len(tokens) - 1):
            # Берем последовательность токенов до текущей позиции
            start_idx = max(0, i - self.sequence_length + 1)
            input_sequence = tokens[start_idx:i + 1]
            
            # Если последовательность короче нужной длины, пропускаем
            if len(input_sequence) < self.sequence_length:
                continue
                
            target_token = tokens[i + 1]
            
            self.samples.append((input_sequence, target_token))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[List[str], str]:
        """Возвращает пример по индексу"""
        return self.samples[idx]


class TokenIndexDataset(Dataset):
    """
    Dataset с числовыми индексами токенов.
    Требует предварительно созданный словарь.
    """
    
    def __init__(self, data_path: str, token_to_idx: dict, sequence_length: int = 10):
        """
        Args:
            data_path: путь к CSV файлу с данными
            token_to_idx: словарь для преобразования токенов в индексы
            sequence_length: длина входной последовательности
        """
        self.sequence_length = sequence_length
        self.token_to_idx = token_to_idx
        self.samples = []
        
        # Загружаем данные
        df = pd.read_csv(data_path)
        
        # Обрабатываем каждую строку
        for _, row in df.iterrows():
            tokenized_text = self._parse_tokenized_text(row['tokenized_text'])
            
            # Преобразуем токены в индексы
            token_indices = [self.token_to_idx.get(token, self.token_to_idx.get('<UNK>', 0)) 
                           for token in tokenized_text]
            
            # Создаем обучающие примеры
            if len(token_indices) > sequence_length:
                for i in range(len(token_indices) - 1):
                    start_idx = max(0, i - self.sequence_length + 1)
                    input_sequence = token_indices[start_idx:i + 1]
                    
                    if len(input_sequence) < self.sequence_length:
                        continue
                        
                    target_token = token_indices[i + 1]
                    self.samples.append((input_sequence, target_token))
    
    def _parse_tokenized_text(self, tokenized_text_str: str) -> List[str]:
        """Парсит строку с токенизированным текстом в список токенов"""
        try:
            return ast.literal_eval(tokenized_text_str)
        except:
            tokens = str(tokenized_text_str).strip("[]").split("', '")
            tokens = [token.strip("'\"") for token in tokens if token.strip("'\"")]
            return tokens
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_sequence, target = self.samples[idx]
        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(target, dtype=torch.long)

#====================================================================
    """Строит словарь с ограничением по размеру"""
    """
    Строит словарь токенов из данных.
    
    Args:
        data_paths: список путей к CSV файлам
        
    Returns:
        token_to_idx: словарь токен -> индекс
        idx_to_token: словарь индекс -> токен
    """
#====================================================================
"""
def build_vocabulary(data_paths: List[str]) -> Tuple[dict, dict]:

    all_tokens = set()
    
    for data_path in data_paths:
        df = pd.read_csv(data_path)
        
        for _, row in df.iterrows():
            try:
                tokens = ast.literal_eval(row['tokenized_text'])
                all_tokens.update(tokens)
            except:
                tokens = str(row['tokenized_text']).strip("[]").split("', '")
                tokens = [token.strip("'\"") for token in tokens if token.strip("'\"")]
                all_tokens.update(tokens)
    
    # Создаем словари
    tokens_list = ['<PAD>', '<UNK>'] + sorted(list(all_tokens))
    token_to_idx = {token: idx for idx, token in enumerate(tokens_list)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    
    print(f"Создан словарь с {len(token_to_idx)} токенами")
    return token_to_idx, idx_to_token
        """
#====================================================================

def build_vocabulary(data_paths: List[str], max_vocab_size: int = 30000) -> Tuple[dict, dict]:
   
    from collections import Counter
    
    token_counter = Counter()
    
    for data_path in data_paths:
        df = pd.read_csv(data_path)
        
        for _, row in df.iterrows():
            try:
                tokens = ast.literal_eval(row['tokenized_text'])
                token_counter.update(tokens)
            except:
                tokens = str(row['tokenized_text']).strip("[]").split("', '")
                tokens = [token.strip("'\"") for token in tokens if token.strip("'\"")]
                token_counter.update(tokens)
    
    # Берем только самые частые токены
    most_common_tokens = token_counter.most_common(max_vocab_size - 2)
    
    # Создаем словари
    tokens_list = ['<PAD>', '<UNK>'] + [token for token, count in most_common_tokens]
    token_to_idx = {token: idx for idx, token in enumerate(tokens_list)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    
    print(f"Создан словарь с {len(token_to_idx)} токенами")
    print(f"Охватывает {sum(count for _, count in most_common_tokens):,} из {sum(token_counter.values()):,} токенов")
    
    return token_to_idx, idx_to_token
#====================================================================

def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 32,
    sequence_length: int = 10,
    shuffle_train: bool = True,
    max_vocab_size: int = 30000  # ⬅️ НОВЫЙ ПАРАМЕТР
) -> Tuple[DataLoader, DataLoader, DataLoader, dict, dict]:
    """
    Создает DataLoader'ы для train, validation и test данных.
    
    Args:
        train_path: путь к train данным
        val_path: путь к validation данным  
        test_path: путь к test данным
        batch_size: размер батча
        sequence_length: длина последовательности
        shuffle_train: перемешивать ли train данные
        max_vocab_size: максимальный размер словаря ⬅️ НОВЫЙ
        
    Returns:
        train_loader, val_loader, test_loader, token_to_idx, idx_to_token
    """
    
    # Строим словарь из всех данных
    token_to_idx, idx_to_token = build_vocabulary([train_path, val_path, test_path], max_vocab_size=max_vocab_size) # ⬅️ ПЕРЕДАЕМ ОГРАНИЧЕНИЕ
    
    # Создаем datasets
    train_dataset = TokenIndexDataset(train_path, token_to_idx, sequence_length)
    val_dataset = TokenIndexDataset(val_path, token_to_idx, sequence_length)
    test_dataset = TokenIndexDataset(test_path, token_to_idx, sequence_length)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Vocabulary size: {len(token_to_idx):,} tokens")  # ⬅️ ДОБАВЬТЕ ЭТО
    
    # Создаем data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader, token_to_idx, idx_to_token


def get_default_data_paths() -> Tuple[str, str, str]:
    """Возвращает пути к данным по умолчанию"""
    project_root = Path(__file__).parent.parent
    train_path = project_root / 'data' / 'train.csv'
    val_path = project_root / 'data' / 'val.csv'
    test_path = project_root / 'data' / 'test.csv'
    
    return str(train_path), str(val_path), str(test_path)


# Функция для быстрого создания data loader'ов
def setup_data_loaders(
    batch_size: int = 32,
    sequence_length: int = 10,
    max_vocab_size: int = 30000  # ⬅️ ДОБАВЬТЕ ПАРАМЕТР
) -> Tuple[DataLoader, DataLoader, DataLoader, dict, dict]:
    """
    Быстрая настройка data loader'ов с путями по умолчанию.
    
    Args:
        batch_size: размер батча
        sequence_length: длина последовательности  
        max_vocab_size: максимальный размер словаря ⬅️ НОВЫЙ
    """
    train_path, val_path, test_path = get_default_data_paths()
    
    return create_data_loaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=batch_size,
        sequence_length=sequence_length,
        max_vocab_size=max_vocab_size  # ⬅️ ПЕРЕДАЕМ ДАЛЬШЕ
    )


# Пример использования
if __name__ == "__main__":
    # Тестируем dataset
    train_path, val_path, test_path = get_default_data_paths()
    
    # Проверяем базовый dataset
    dataset = NextTokenDataset(train_path, sequence_length=5)
    print(f"Dataset size: {len(dataset)}")
    
    # Показываем несколько примеров
    for i in range(3):
        x, y = dataset[i]
        print(f"Пример {i}: X={x} -> Y={y}")
    
    # Создаем data loader'ы
    train_loader, val_loader, test_loader, token_to_idx, idx_to_token = setup_data_loaders(
        batch_size=16,
        sequence_length=10
    )
    
    # Проверяем первый батч
    for x_batch, y_batch in train_loader:
        print(f"Размер батча: X={x_batch.shape}, Y={y_batch.shape}")
        print(f"Пример X: {x_batch[0]}")
        print(f"Пример Y: {y_batch[0]}")
        break