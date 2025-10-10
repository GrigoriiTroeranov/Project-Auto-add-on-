# src/data_utils.py

import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
import nltk

def setup_nltk():
    """Настройка NLTK с обработкой ошибок"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Скачивание токенизатора NLTK...")
        nltk.download('punkt', quiet=True)

def clean_text(text):
    """
    Очистка и нормализация текста.
    """
    if not isinstance(text, str):
        return ""
    
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Удаление упоминаний пользователей (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Удаление специальных символов (оставляем только буквы, цифры и основные знаки препинания)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    
    # Удаление повторяющихся пробелов
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def simple_tokenize(text):
    """
    Простая токенизация без NLTK
    """
    if not text or text.strip() == "":
        return []
    return text.split()

def process_dataset():
    """
    Основная функция обработки датасета.
    """
    # Настраиваем NLTK
    setup_nltk()
    
    # Определяем пути
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / 'data' / 'raw_dataset.csv'
    processed_data_path = project_root / 'data' / 'dataset_processed.csv'
    
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Исходный файл не найден: {raw_data_path}")
    
    print("Чтение исходного датасета...")
    
    # Читаем CSV файл
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(raw_data_path, encoding='latin-1', names=column_names)
    
    print(f"Загружено {len(df)} записей")
    
    # Очищаем текст
    print("Очистка текста...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Удаляем пустые тексты
    initial_count = len(df)
    df = df[df['cleaned_text'].str.len() > 0]
    final_count = len(df)
    
    print(f"Удалено {initial_count - final_count} пустых записей")
    print(f"Осталось {final_count} записей")
    
    # Токенизируем текст (простой метод)
    print("Токенизация текста...")
    df['tokenized_text'] = df['cleaned_text'].apply(simple_tokenize)
    
    # Сохраняем обработанный датасет
    processed_df = df[['cleaned_text', 'tokenized_text']]
    processed_df.to_csv(processed_data_path, index=False, encoding='utf-8')
    
    print(f"Обработанный датасет сохранен: {processed_data_path}")
    
    return processed_df

def split_dataset(processed_df=None, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Разделяет dataset на train, validation и test наборы.
    """
    if processed_df is None:
        processed_data_path = Path(__file__).parent.parent / 'data' / 'dataset_processed.csv'
        if not processed_data_path.exists():
            raise FileNotFoundError(f"Обработанный датасет не найден: {processed_data_path}")
        processed_df = pd.read_csv(processed_data_path, encoding='utf-8')
    
    project_root = Path(__file__).parent.parent
    train_path = project_root / 'data' / 'train.csv'
    val_path = project_root / 'data' / 'val.csv'
    test_path = project_root / 'data' / 'test.csv'
    
    print("Разделение датасета...")
    
    # Разделяем на train и временный набор
    train_df, temp_df = train_test_split(
        processed_df, 
        train_size=train_size, 
        random_state=random_state,
        shuffle=True
    )
    
    # Разделяем временный набор на val и test
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        random_state=random_state,
        shuffle=True
    )
    
    # Сохраняем
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train: {len(train_df)} записей")
    print(f"Val: {len(val_df)} записей")
    print(f"Test: {len(test_df)} записей")
    
    return train_df, val_df, test_df

def main():
    """
    Основная функция для выполнения всего пайплайна.
    """
    print("Запуск обработки данных...")
    processed_df = process_dataset()
    train_df, val_df, test_df = split_dataset(processed_df)
    print("Обработка данных завершена!")

if __name__ == "__main__":
    main()