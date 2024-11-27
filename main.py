import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import random
import time
import threading
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Toplevel, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Ініціалізація головного вікна
root = tk.Tk()
root.title("SepSimTDA: No file loaded")
root.resizable(False, False)
root.attributes('-toolwindow', True)  # Прибираємо системне меню, що включає restore, move, size, minimize, maximize

# Флаг для зупинки потоку
stop_thread = threading.Event()

audiofile_y = None
audiofile_sr = None

def plot_spectrogram(y, sr, n_fft=2048, hop_length=512, figsize=(10, 6)):
    """
    # Обчислення спектрограми (STFT)
    n_fft # Оптимальний розмір вікна для більш точної частотної роздільної здатності
    hop_length # Крок для перекриття вікон
    """
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=np.hanning(n_fft))
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    S_db += np.abs(S_db.min())

    # Створення фігури для відображення спектрограми
    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='viridis', ax=ax)
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB', ticks=np.linspace(S_db.min(), S_db.max(), 10), orientation='vertical')
    cbar.set_label('Power (positive dB)')
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel('Time (s)')
    ax.set_xticks(np.linspace(0, S_db.shape[1] * hop_length / sr, 10))
    ax.set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    return fig

def truncate_text(text, max_length):
    """ Обрізає текст і додає '...' у разі перевищення максимального розміру. """
    if len(text) > max_length:
        return text[:max_length - 3] + "..."
    return text

def load_mix():
    """ Функція для вибору wav-файлу. """
    filepath = filedialog.askopenfilename(
        filetypes=[("WAV files", "*.wav")],
        title="Select a WAV file"
    )

    progress_bar['value'] = 0  # Скидання значення прогрес-бару в нуль
    if filepath:
        # Заблокувати кнопки Source і Show
        for button in source_buttons:
            button.config(state="disabled")

        global loaded_filename
        # Оновлення назви файлу в текстовому полі
        loaded_filename.set(truncate_text(filepath, 24))  # Обрізаємо до 24 символів
        root.title(f"SepSimTDA: {loaded_filename.get()}")  # Оновлення заголовку вікнаОновлення заголовку вікна
        # Розблокувати кнопку SeparateMix
        separate_button.config(state="normal")
        
        global audiofile_y, audiofile_sr
        audiofile_y, audiofile_sr = librosa.load(filepath, sr=None)


def save_source(index):
    """ Функція, що викликається при натисканні кнопки Source. """
    print(f"SaveSource called with index: {index}")
    save_path = filedialog.asksaveasfilename(
        defaultextension=".wav",
        filetypes=[("WAV files", "*.wav")],
        title="Save Source File"
    )
    if save_path:
        print(f"File saved to: {save_path}")

def spectrogram(index):
    global audiofile_y, audiofile_sr

    """ Функція, що викликається при натисканні кнопки Spectrogram1. """
    print(f"Spectrogram called with index: {index}")

    fig = plot_spectrogram(audiofile_y, audiofile_sr, figsize=(8, 4))

    # Створення нового вікна для відображення спектрограми
    spectrogram_window = Toplevel()
    spectrogram_window.title(f"SepSimTDA: Voice{index}")
    spectrogram_window.resizable(False, False)
    spectrogram_window.attributes('-toolwindow', True)

    # Відображення в новому вікні через tkinter
    canvas = FigureCanvasTkAgg(fig, master=spectrogram_window)
    canvas.get_tk_widget().pack()
    canvas.draw()


def audio_spectrogram():
    spectrogram(0) # TODO: temp

def separate_mix():
    """ Функція для створення випадкової кількості кнопок з симуляцією затримки. """
    # Заблокувати кнопки LoadMix, SeparateMix і пари Source та Show
    load_button.config(state="disabled")
    separate_button.config(state="disabled")
    for button in source_buttons:
        button.config(state="disabled")

    # Показати прогрес-бар і встановити його на 0%
    progress_bar['value'] = 0
    root.update_idletasks()

    def process_separation():
        # Симуляція затримки
        for i in range(101):
            if stop_thread.is_set():
                return
            progress_bar['value'] = i
            root.update_idletasks()
            time.sleep(0.02)  # Оновлення кожні 0.02 секунди

        # Завершити прогрес-бар на 100%
        root.update_idletasks()
        progress_bar['value'] = 100

        # Розблокувати кнопку LoadMix і пари Source та Show
        load_button.config(state="normal")
        for button in source_buttons:
            button.config(state="normal")

    # Запуск процесу розділення в окремому потоці
    threading.Thread(target=process_separation, daemon=True).start()



# Список для збереження кнопок Source
source_buttons = []

# Перемінна для збереження назви файлу
loaded_filename = tk.StringVar(value="NoFileLoaded")

# Рядок для кнопок LoadMix і MixSpectrogram
button_frame = tk.Frame(root)
button_frame.pack(side="top", fill="x", padx=10, pady=10)

# Створення кнопки LoadMix
load_button = tk.Button(button_frame, text="LoadAudio", command=load_mix)
load_button.pack(side="left", padx=5)

# Створення кнопки MixSpectrogram
mix_spectrogram_button = tk.Button(button_frame, text="ShowSpectrogram", command=audio_spectrogram)
mix_spectrogram_button.pack(side="left", padx=5)

# Кнопка SeparateMix і progress_bar в третьому рядку
separate_frame = tk.Frame(root)
separate_frame.pack(side="top", fill="x", padx=10, pady=10)

# Створення кнопки SeparateMix (спочатку заблокована)
separate_button = tk.Button(separate_frame, text="Separation", command=separate_mix, state="disabled")
separate_button.pack(side="left", padx=5)

# Прогрес-бар для відображення процесу розділення (завжди показаний праворуч від кнопки SeparateMix)
progress_bar = ttk.Progressbar(separate_frame, orient="horizontal", mode="determinate", maximum=100)
progress_bar.pack(side="left", fill="x", expand=True, padx=10, pady=5)

# Створення 4 пар кнопок Source і Show
for i in range(1, 5):
    button_frame = tk.Frame(root)
    button_frame.pack(side="top", anchor="w", pady=2, padx=10)
    button = tk.Button(button_frame, text=f"LoadVoice{i}", command=lambda i=i: save_source(i), state="disabled")
    button.pack(side="left", padx=5)
    
    # Додавання кнопки Show
    spectrogram_button = tk.Button(button_frame, text="ShowSpectrogram", command=lambda i=i: spectrogram(i), state="disabled")
    spectrogram_button.pack(side="left", padx=5)
    
    source_buttons.append(button)
    source_buttons.append(spectrogram_button)

# Обробник для закриття вікна
def on_closing():
    stop_thread.set()  # Встановити флаг зупинки потоку
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Запуск головного циклу програми
root.mainloop()