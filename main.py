# import tkinter as tk
# from tkinter import filedialog
# from tkinter import ttk
# import random
# import time
# import threading

# # Ініціалізація головного вікна
# root = tk.Tk()
# root.title("SepSimTDA")
# root.resizable(False, False)
# root.attributes('-toolwindow', True)  # Прибираємо системне меню, що включає restore, move, size, minimize, maximize

# # Флаг для зупинки потоку
# stop_thread = threading.Event()

# def truncate_text(text, max_length):
#     """ Обрізає текст і додає '...' у разі перевищення максимального розміру. """
#     if len(text) > max_length:
#         return text[:max_length - 3] + "..."
#     return text

# def load_mix():
#     """ Функція для вибору wav-файлу. """
#     filepath = filedialog.askopenfilename(
#         filetypes=[("WAV files", "*.wav")],
#         title="Select a WAV file"
#     )
#     if filepath:
#         global loaded_filename
#         # Оновлення назви файлу в текстовому полі
#         loaded_filename.set(truncate_text(filepath, 24))  # Обрізаємо до 50 символів
#         # Розблокувати кнопку SeparateMix
#         separate_button.config(state="normal")
#         # Оновити розмір вікна
#         update_window_size()

# def save_source(index):
#     """ Функція, що викликається при натисканні кнопки Source. """
#     print(f"SaveSource called with index: {index}")
#     save_path = filedialog.asksaveasfilename(
#         defaultextension=".wav",
#         filetypes=[("WAV files", "*.wav")],
#         title="Save Source File"
#     )
#     if save_path:
#         print(f"File saved to: {save_path}")

# def separate_mix():
#     """ Функція для створення випадкової кількості кнопок з симуляцією затримки. """
#     # Заблокувати кнопки LoadMix і SeparateMix
#     load_button.config(state="disabled")
#     separate_button.config(state="disabled")

#     # Видалення старих кнопок, якщо такі є
#     for button in source_buttons:
#         button.destroy()
#     source_buttons.clear()

#     # Показати прогрес-бар і встановити його на 0%
#     progress_bar.pack(side="top", fill="x", padx=10, pady=5)
#     progress_bar['value'] = 0
#     root.update_idletasks()

#     def process_separation():
#         # Симуляція затримки
#         for i in range(101):
#             if stop_thread.is_set():
#                 return
#             progress_bar['value'] = i
#             root.update_idletasks()
#             time.sleep(0.02)  # Оновлення кожні 0.02 секунди

#         # Створення нових кнопок
#         num_sources = 2
#         for i in range(1, num_sources + 1):
#             if stop_thread.is_set():
#                 return
#             button = tk.Button(root, text=f"Source{i}", command=lambda i=i: save_source(i))
#             button.pack(side="top", anchor="w", pady=2, padx=10, fill="x")
#             source_buttons.append(button)

#         # Завершити прогрес-бар на 100% і сховати його
#         root.update_idletasks()
#         progress_bar.pack_forget()

#         # Розблокувати кнопку LoadMix
#         load_button.config(state="normal")

#         # Оновити розмір вікна
#         update_window_size()

#     # Запуск процесу розділення в окремому потоці
#     threading.Thread(target=process_separation, daemon=True).start()

# def update_window_size():
#     """ Оновлює розмір вікна залежно від розмірів усіх віджетів. """
#     # Отримати ширину кнопок LoadMix і SeparateMix
#     button_frame_width = sum(button.winfo_reqwidth() for button in button_frame.winfo_children()) + 40

#     # Отримати висоту кнопки LoadMix, текстового поля і кнопок Source
#     total_height = (
#         load_button.winfo_reqheight()  # Висота кнопки LoadMix
#         + filename_label.winfo_reqheight()  # Висота текстового поля
#         + sum(button.winfo_reqheight() for button in source_buttons)  # Висота кнопок Source
#         + (progress_bar.winfo_reqheight() if progress_bar.winfo_ismapped() else 0)  # Висота прогрес-бару, якщо він відображається
#         + 60  # Відступи
#     )

#     # Встановити новий розмір вікна
#     root.geometry(f"{button_frame_width}x{total_height}")

# # Список для збереження кнопок Source
# source_buttons = []

# # Перемінна для збереження назви файлу
# loaded_filename = tk.StringVar(value="No file loaded")

# # Рядок для кнопок LoadMix і SeparateMix
# button_frame = tk.Frame(root)
# button_frame.pack(side="top", fill="x", padx=10, pady=10)

# # Створення кнопки LoadMix
# load_button = tk.Button(button_frame, text="LoadMix", command=load_mix)
# load_button.pack(side="left", padx=5)

# # Створення кнопки SeparateMix (спочатку заблокована)
# separate_button = tk.Button(button_frame, text="SeparateMix", command=separate_mix, state="disabled")
# separate_button.pack(side="left", padx=5)

# # Поле для відображення назви файлу
# filename_label = tk.Label(root, textvariable=loaded_filename, anchor="w", wraplength=580, justify="left")
# filename_label.pack(side="top", fill="x", padx=10, pady=5)

# # Прогрес-бар для відображення процесу розділення
# progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate", maximum=100)

# # Оновлення початкового розміру вікна
# root.update_idletasks()  # Оновити інформацію про розміри віджетів
# update_window_size()

# # Обробник для закриття вікна
# def on_closing():
#     stop_thread.set()  # Встановити флаг зупинки потоку
#     root.destroy()

# root.protocol("WM_DELETE_WINDOW", on_closing)

# # Запуск головного циклу програми
# root.mainloop()

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog, Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_spectrogram():
    # Вибір аудіофайлу через діалогове вікно
    audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if not audio_path:
        return

    # Завантаження аудіофайлу
    y, sr = librosa.load(audio_path, sr=None)  # Використовуємо оригінальну частоту дискретизації

    # Обчислення спектрограми (STFT)
    n_fft = 2048  # Оптимальний розмір вікна для більш точної частотної роздільної здатності
    hop_length = 512  # Крок для перекриття вікон
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=np.hanning(n_fft))
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    S_db += np.abs(S_db.min())

    # Створення нового вікна для відображення спектрограми
    spectrogram_window = Toplevel()
    spectrogram_window.geometry('1200x800')
    spectrogram_window.title("Spectrogram")

    # Відображення спектрограми
    fig, ax = plt.subplots(figsize=(10, 6))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='viridis', ax=ax)
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB', ticks=np.linspace(S_db.min(), S_db.max(), 10), orientation='vertical')
    cbar.ax.invert_yaxis()
    cbar.set_label('Power (positive dB)')
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel('Time (s)')
    ax.set_xticks(np.linspace(0, S_db.shape[1] * hop_length / sr, 10))
    ax.set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()

    # Відображення в новому вікні через tkinter
    canvas_frame = tk.Frame(spectrogram_window)
    canvas_frame.pack(fill=tk.BOTH, expand=True)

    h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    h_scrollbar.config(command=canvas.get_tk_widget().xview)
    canvas.draw()

# Створення графічного інтерфейсу користувача
root = tk.Tk()
root.title("Spectrogram Viewer")
root.geometry("300x150")

# Додавання кнопки для відображення спектрограми
button = tk.Button(root, text="Open Audio File and Plot Spectrogram", command=plot_spectrogram)
button.pack(expand=True)

root.protocol("WM_DELETE_WINDOW", root.quit)
root.mainloop()
