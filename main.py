import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import random
import time
import threading

# Ініціалізація головного вікна
root = tk.Tk()
root.title("")
root.resizable(False, False)

def truncate_text(text, max_length):
    """Обрізає текст і додає '...' у разі перевищення максимального розміру."""
    if len(text) > max_length:
        return text[:max_length - 3] + "..."
    return text


def load_mix():
    """Функція для вибору wav-файлу."""
    filepath = filedialog.askopenfilename(
        filetypes=[("WAV files", "*.wav")],
        title="Select a WAV file"
    )
    if filepath:
        global loaded_filename
        # Оновлення назви файлу в текстовому полі
        loaded_filename.set(truncate_text(filepath, 24))  # Обрізаємо до 50 символів
        # Розблокувати кнопку SeparateMix
        separate_button.config(state="normal")
        # Оновити розмір вікна
        update_window_size()


def save_source(index):
    """Функція, що викликається при натисканні кнопки Source."""
    print(f"SaveSource called with index: {index}")
    save_path = filedialog.asksaveasfilename(
        defaultextension=".wav",
        filetypes=[("WAV files", "*.wav")],
        title="Save Source File"
    )
    if save_path:
        print(f"File saved to: {save_path}")


def separate_mix():
    """Функція для створення випадкової кількості кнопок з симуляцією затримки."""
    # Заблокувати кнопки LoadMix і SeparateMix
    load_button.config(state="disabled")
    separate_button.config(state="disabled")

    # Видалення старих кнопок, якщо такі є
    for button in source_buttons:
        button.destroy()
    source_buttons.clear()

    # Показати прогрес-бар і встановити його на 0%
    progress_bar.pack(side="top", fill="x", padx=10, pady=5)
    progress_bar['value'] = 0
    root.update_idletasks()

    def process_separation():
        # Симуляція затримки
        time.sleep(2)  # Затримка 2 секунди для імітації процесу розділення

        # Імітація процесу для демонстрації роботи прогрес-бару
        progress_bar['value'] = 50
        root.update_idletasks()

        # Створення нових кнопок
        num_sources = 2
        for i in range(1, num_sources + 1):
            button = tk.Button(root, text=f"Source{i}", command=lambda i=i: save_source(i))
            button.pack(side="top", anchor="w", pady=2, padx=10, fill="x")
            source_buttons.append(button)

        # Завершити прогрес-бар на 100% і сховати його
        progress_bar['value'] = 100
        root.update_idletasks()
        time.sleep(1)  # Затримка для показу завершеного прогрес-бару
        progress_bar.pack_forget()

        # Розблокувати кнопку LoadMix
        load_button.config(state="normal")

        # Оновити розмір вікна
        update_window_size()

    # Запуск процесу розділення в окремому потоці
    threading.Thread(target=process_separation).start()


def update_window_size():
    """Оновлює розмір вікна залежно від розмірів усіх віджетів."""
    # Отримати ширину кнопок LoadMix і SeparateMix
    button_frame_width = sum(button.winfo_reqwidth() for button in button_frame.winfo_children()) + 40

    # Отримати висоту кнопки LoadMix, текстового поля і кнопок Source
    total_height = (
        load_button.winfo_reqheight()  # Висота кнопки LoadMix
        + filename_label.winfo_reqheight()  # Висота текстового поля
        + sum(button.winfo_reqheight() for button in source_buttons)  # Висота кнопок Source
        + (progress_bar.winfo_reqheight() if progress_bar.winfo_ismapped() else 0)  # Висота прогрес-бару, якщо він відображається
        + 60  # Відступи
    )

    # Встановити новий розмір вікна
    root.geometry(f"{button_frame_width}x{total_height}")

# Список для збереження кнопок Source
source_buttons = []

# Перемінна для збереження назви файлу
loaded_filename = tk.StringVar(value="No file loaded")

# Рядок для кнопок LoadMix і SeparateMix
button_frame = tk.Frame(root)
button_frame.pack(side="top", fill="x", padx=10, pady=10)

# Створення кнопки LoadMix
load_button = tk.Button(button_frame, text="LoadMix", command=load_mix)
load_button.pack(side="left", padx=5)

# Створення кнопки SeparateMix (спочатку заблокована)
separate_button = tk.Button(button_frame, text="SeparateMix", command=separate_mix, state="disabled")
separate_button.pack(side="left", padx=5)

# Поле для відображення назви файлу
filename_label = tk.Label(root, textvariable=loaded_filename, anchor="w", wraplength=580, justify="left")
filename_label.pack(side="top", fill="x", padx=10, pady=5)

# Прогрес-бар для відображення процесу розділення
progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate", maximum=100)

# Оновлення початкового розміру вікна
root.update_idletasks()  # Оновити інформацію про розміри віджетів
update_window_size()

# Запуск головного циклу програми
root.mainloop()