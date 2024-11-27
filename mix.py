from pydub import AudioSegment
from pydub.utils import which

# Вказуємо шлях до FFmpeg, якщо виникає проблема
AudioSegment.converter = which("ffmpeg")

# Завантажуємо два wav файли
audio1 = AudioSegment.from_wav("0.wav")
audio2 = AudioSegment.from_wav("1.wav")

# Знаходимо мінімальну довжину
min_length = min(len(audio1), len(audio2))

# Обрізаємо обидва аудіо до найкоротшої тривалості
audio1 = audio1[:min_length]
audio2 = audio2[:min_length]

# Перезаписуємо обрізані аудіо назад у вихідні файли
audio1.export("0.wav", format="wav")
audio2.export("1.wav", format="wav")

# Накладаємо один файл на інший
combined_audio = audio1.overlay(audio2)

# Зберігаємо результат у новий файл
combined_audio.export("01.wav", format="wav")
