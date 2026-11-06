"""Russian desktop interface. No browser, API keys, or photo uploads."""
import logging
import queue
import secrets
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk, PngImagePlugin
from engine import HeadSwapEngine, Cancelled
from imaging import Box, open_photo, validate_boxes

ROOT = Path(__file__).resolve().parent
(ROOT / 'logs').mkdir(exist_ok=True)
if sys.stdout is None:
    sys.stdout = (ROOT / 'logs' / 'runtime.log').open('a', encoding='utf-8', buffering=1)
if sys.stderr is None:
    sys.stderr = sys.stdout
logging.basicConfig(filename=ROOT / 'logs' / 'app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8')
BG, PANEL, TEXT, MUTED, ACCENT = '#11151f', '#1b2230', '#edf2fa', '#a7b4c8', '#8cb5ff'
COLORS = ['#8cb5ff', '#ffbc85']


class PhotoCanvas(tk.Canvas):
    def __init__(self, parent, on_change):
        super().__init__(parent, bg='#0b0e15', highlightthickness=0, cursor='crosshair')
        self.photo = None
        self.boxes = []
        self.editable = True
        self.limit = 2
        self.scale = 1
        self.offset = (0, 0)
        self.anchor = None
        self.preview = None
        self.on_change = on_change
        self.bind('<Configure>', lambda e: self.redraw())
        self.bind('<ButtonPress-1>', self.begin)
        self.bind('<B1-Motion>', self.move)
        self.bind('<ButtonRelease-1>', self.finish)
        self.bind('<ButtonPress-3>', lambda e: self.undo())

    def set_image(self, image):
        self.photo = image
        self.redraw()

    def image_point(self, x, y):
        if self.photo is None:
            return 0, 0
        ox, oy = self.offset
        return (round(max(0, min(self.photo.width, (x - ox) / self.scale))),
                round(max(0, min(self.photo.height, (y - oy) / self.scale))))

    def begin(self, event):
        if self.editable and self.photo and len(self.boxes) < self.limit:
            self.anchor = self.image_point(event.x, event.y)
            self.preview = None

    def move(self, event):
        if self.anchor is not None:
            x, y = self.image_point(event.x, event.y)
            ax, ay = self.anchor
            self.preview = Box(min(ax, x), min(ay, y), max(ax, x), max(ay, y))
            self.redraw()

    def finish(self, event):
        if self.anchor is not None:
            self.move(event)
            if self.preview and min(self.preview.width, self.preview.height) >= 24:
                self.boxes.append(self.preview)
            self.anchor = self.preview = None
            self.redraw()
            self.on_change()

    def undo(self):
        if self.editable and self.boxes:
            self.boxes.pop()
            self.redraw()
            self.on_change()

    def clear(self):
        self.boxes.clear()
        self.anchor = self.preview = None
        self.redraw()
        self.on_change()

    def redraw(self):
        self.delete('all')
        w, h = max(1, self.winfo_width()), max(1, self.winfo_height())
        if self.photo is None:
            self.create_text(w / 2, h / 2, text='Открой фотографию, чтобы начать',
                             fill=MUTED, font=('Segoe UI', 15))
            return
        self.scale = min((w - 24) / self.photo.width, (h - 24) / self.photo.height)
        self.scale = max(self.scale, 0.001)
        sw, sh = max(1, round(self.photo.width * self.scale)), max(1, round(self.photo.height * self.scale))
        self.offset = ((w - sw) / 2, (h - sh) / 2)
        self.display_image = ImageTk.PhotoImage(self.photo.resize((sw, sh), Image.Resampling.LANCZOS))
        ox, oy = self.offset
        self.create_image(ox, oy, image=self.display_image, anchor='nw')
        if not self.editable:
            return
        for index, box in enumerate(self.boxes + ([self.preview] if self.preview else [])):
            color = COLORS[min(index, 1)]
            coords = (ox + box.x1 * self.scale, oy + box.y1 * self.scale,
                      ox + box.x2 * self.scale, oy + box.y2 * self.scale)
            self.create_rectangle(*coords, outline=color, width=2, dash=(6, 3))
            self.create_rectangle(coords[0], coords[1], coords[0] + 28, coords[1] + 25,
                                  fill=color, outline=color)
            self.create_text(coords[0] + 14, coords[1] + 12, text='А' if index == 0 else 'Б',
                             fill=BG, font=('Segoe UI', 11, 'bold'))


class Application:
    def __init__(self, root):
        self.root = root
        root.title('Головы местами · HeadSwap')
        root.geometry('1180x820')
        root.minsize(940, 660)
        root.configure(bg=BG)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.engine = HeadSwapEngine()
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.closing = False
        self.original = self.reference = self.result = None
        self.source_path = None
        self.seed = secrets.randbelow(2**31 - 1)
        self.result_seed = None
        self.mode = tk.StringVar(value='Обмен на одном фото')
        self.quality = tk.StringVar(value='Быстро · 512')
        self.view = tk.StringVar(value='Исходник')
        self.status = tk.StringVar(value='Открой фото и отметь две головы прямоугольниками.')
        self.count = tk.StringVar(value='Выделено 0 из 2')
        self.style()
        self.build()
        root.after(100, self.poll)

    def style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=BG, foreground=TEXT, font=('Segoe UI', 10))
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=TEXT)
        style.configure('Muted.TLabel', foreground=MUTED)
        style.configure('Title.TLabel', font=('Segoe UI', 23, 'bold'))
        style.configure('TButton', background=PANEL, foreground=TEXT, padding=(12, 9), borderwidth=0)
        style.map('TButton', background=[('active', '#2b3850'), ('disabled', '#19202b')],
                  foreground=[('disabled', '#667185')])
        style.configure('Accent.TButton', background=ACCENT, foreground=BG, font=('Segoe UI', 11, 'bold'))
        style.map('Accent.TButton', background=[('active', '#b0ccff'), ('disabled', '#2d3a50')])
        style.configure('TCombobox', fieldbackground=PANEL, background=PANEL,
                        foreground=TEXT, arrowcolor=TEXT, padding=7)
        style.map('TCombobox', fieldbackground=[('readonly', PANEL)], foreground=[('readonly', TEXT)])
        style.configure('TRadiobutton', background=BG, foreground=TEXT, padding=5)
        style.configure('Horizontal.TProgressbar', background=ACCENT, troughcolor=PANEL, borderwidth=0)

    def build(self):
        shell = ttk.Frame(self.root, padding=24)
        shell.pack(fill='both', expand=True)
        header = ttk.Frame(shell)
        header.pack(fill='x')
        ttk.Label(header, text='Головы местами', style='Title.TLabel').pack(side='left')
        ttk.Label(header, text='●  На твоём компьютере · без облака', foreground='#9ed8b7').pack(side='right')
        ttk.Label(shell, text='Лицо, волосы и форма головы. Одежда, поза и фон за пределами выделения сохраняются.',
                  style='Muted.TLabel').pack(anchor='w', pady=(5, 18))
        toolbar = ttk.Frame(shell)
        toolbar.pack(fill='x')
        self.open_button = ttk.Button(toolbar, text='Открыть фото…', command=self.open_main)
        self.open_button.pack(side='left', padx=(0, 12))
        self.mode_box = ttk.Combobox(toolbar, textvariable=self.mode, state='readonly', width=29,
                                    values=['Обмен на одном фото', 'Голова с другого фото'])
        self.mode_box.pack(side='left')
        self.mode_box.bind('<<ComboboxSelected>>', self.change_mode)
        self.reference_button = ttk.Button(toolbar, text='Фото с нужной головой…', command=self.open_reference, state='disabled')
        self.reference_button.pack(side='left', padx=12)
        self.clear_button = ttk.Button(toolbar, text='Сбросить выделение', command=self.clear)
        self.clear_button.pack(side='right')
        info = ttk.Frame(shell)
        info.pack(fill='x', pady=(14, 8))
        self.instructions = ttk.Label(info, text='Обведи голову А, затем Б — целиком, с волосами и небольшим запасом.',
                                      style='Muted.TLabel')
        self.instructions.pack(side='left')
        ttk.Label(info, textvariable=self.count, foreground=ACCENT).pack(side='right')
        content = ttk.Frame(shell)
        content.pack(fill='both', expand=True)
        self.canvas = PhotoCanvas(content, self.selection_changed)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.reference_panel = ttk.Frame(content, padding=(14, 0, 0, 0), width=190)
        ttk.Label(self.reference_panel, text='Новая голова', font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(0, 8))
        self.reference_label = ttk.Label(self.reference_panel, text='Загрузи портрет\nодного человека',
                                         style='Muted.TLabel', justify='left', wraplength=175)
        self.reference_label.pack(anchor='n')
        ttk.Label(self.reference_panel, text='Лучше крупный портрет:\nголова целиком, без обрезанных волос.',
                  style='Muted.TLabel', wraplength=175).pack(anchor='w', pady=12)
        bottom = ttk.Frame(shell)
        bottom.pack(fill='x', pady=(12, 0))
        self.before = ttk.Radiobutton(bottom, text='Исходник', variable=self.view, value='Исходник', command=self.change_view)
        self.before.pack(side='left')
        self.after = ttk.Radiobutton(bottom, text='Результат', variable=self.view, value='Результат', command=self.change_view, state='disabled')
        self.after.pack(side='left', padx=(0, 14))
        self.quality_box = ttk.Combobox(bottom, textvariable=self.quality, state='readonly', width=17,
                                       values=['Быстро · 512', 'Детальнее · 768'])
        self.quality_box.pack(side='left')
        self.save_button = ttk.Button(bottom, text='Сохранить PNG…', command=self.save, state='disabled')
        self.save_button.pack(side='right')
        self.run_button = ttk.Button(bottom, text='Поменять головы', command=self.run, style='Accent.TButton', state='disabled')
        self.run_button.pack(side='right', padx=10)
        self.cancel_button = ttk.Button(bottom, text='Остановить', command=self.stop, state='disabled')
        self.cancel_button.pack(side='right')
        self.bar = ttk.Progressbar(shell, mode='determinate')
        self.bar.pack(fill='x', pady=(14, 9))
        ttk.Label(shell, textvariable=self.status, style='Muted.TLabel', wraplength=1080).pack(anchor='w')
        ttk.Label(shell, text='Правая кнопка мыши отменяет последнее выделение. Повторный запуск создаёт другой вариант.',
                  style='Muted.TLabel', font=('Segoe UI', 9)).pack(anchor='w', pady=(5, 0))

    def open_main(self):
        path = filedialog.askopenfilename(title='Исходная фотография', filetypes=[('Фотографии', '*.jpg *.jpeg *.png *.webp *.bmp'), ('Все файлы', '*.*')])
        if not path:
            return
        self.load_main(path)

    def load_main(self, path):
        try:
            self.original = open_photo(path)
        except Exception as error:
            messagebox.showerror('Не удалось открыть фото', str(error))
            return
        self.source_path = Path(path)
        self.result = None
        self.result_seed = None
        self.view.set('Исходник')
        self.canvas.set_image(self.original)
        self.canvas.clear()
        self.after.configure(state='disabled')
        self.save_button.configure(state='disabled')
        self.change_view()

    def open_reference(self):
        path = filedialog.askopenfilename(title='Фото с нужной головой', filetypes=[('Фотографии', '*.jpg *.jpeg *.png *.webp *.bmp'), ('Все файлы', '*.*')])
        if not path:
            return
        try:
            self.reference = open_photo(path)
        except Exception as error:
            messagebox.showerror('Не удалось открыть фото', str(error))
            return
        preview = self.reference.copy()
        preview.thumbnail((175, 240), Image.Resampling.LANCZOS)
        self.reference_preview = ImageTk.PhotoImage(preview)
        self.reference_label.configure(image=self.reference_preview, text='')
        self.selection_changed()

    def change_mode(self, event=None):
        mutual = self.mode.get() == 'Обмен на одном фото'
        self.canvas.limit = 2 if mutual else 1
        self.reference_button.configure(state='disabled' if mutual else 'normal')
        self.reference_panel.pack_forget()
        if not mutual:
            self.reference_panel.pack(side='right', fill='y')
        self.instructions.configure(text=('Обведи голову А, затем Б — целиком, с волосами и небольшим запасом.' if mutual
                                          else 'Обведи голову, которую заменить. Справа загрузи портрет для переноса.'))
        self.run_button.configure(text='Поменять головы' if mutual else 'Заменить голову')
        self.clear()

    def clear(self):
        self.view.set('Исходник')
        self.change_view()
        self.canvas.clear()

    def selection_changed(self):
        count = self.canvas.limit
        self.count.set(f'Выделено {len(self.canvas.boxes)} из {count}')
        ready = self.original is not None and len(self.canvas.boxes) == count
        ready = ready and (count == 2 or self.reference is not None) and not self.busy
        self.run_button.configure(state='normal' if ready else 'disabled')
        if not self.busy:
            self.status.set('Готово к обработке. В выделение должна помещаться и новая форма головы.' if ready
                            else 'Открой фото, выдели головы целиком и добавь портрет, если нужен перенос.')

    def change_view(self):
        show_result = self.view.get() == 'Результат' and self.result is not None
        self.canvas.editable = not show_result and not self.busy
        self.canvas.set_image(self.result if show_result else self.original)

    def set_busy(self, busy):
        self.busy = busy
        for button in (self.open_button, self.clear_button):
            button.configure(state='disabled' if busy else 'normal')
        self.reference_button.configure(state='disabled' if busy or self.canvas.limit == 2 else 'normal')
        for combo in (self.mode_box, self.quality_box):
            combo.configure(state='disabled' if busy else 'readonly')
        self.cancel_button.configure(state='normal' if busy else 'disabled')
        self.before.configure(state='disabled' if busy else 'normal')
        self.after.configure(state='normal' if not busy and self.result else 'disabled')
        self.save_button.configure(state='normal' if not busy and self.result else 'disabled')
        self.canvas.editable = not busy and self.view.get() == 'Исходник'
        self.selection_changed()
        self.canvas.redraw()
        if busy:
            self.bar.configure(mode='indeterminate')
            self.bar.start(12)
        else:
            self.bar.stop()
            self.bar.configure(mode='determinate', value=0)

    def run(self):
        if self.busy or self.original is None:
            return
        try:
            validate_boxes(self.canvas.boxes, self.original.size, self.canvas.limit)
        except ValueError as error:
            messagebox.showerror('Проверь выделение', str(error))
            return
        self.cancel.clear()
        self.seed = secrets.randbelow(2**31 - 1)
        self.set_busy(True)
        self.started = time.monotonic()
        self.status.set('Подготавливаю обработку…')
        photo = self.original.copy()
        boxes = list(self.canvas.boxes)
        reference = self.reference.copy() if self.canvas.limit == 1 else None
        resolution = 512 if self.quality.get().endswith('512') else 768
        seed = self.seed

        def worker():
            try:
                output = self.engine.swap(photo, boxes, reference, resolution, seed,
                                          lambda text: self.events.put(('progress', text)), self.cancel)
                self.events.put(('done', (output, seed)))
            except Cancelled:
                self.events.put(('cancelled', None))
            except Exception as error:
                logging.exception('Generation failed')
                text = str(error)
                if 'out of memory' in text.lower():
                    text = 'Не хватает видеопамяти. Закрой игры и Unity, выбери «Быстро · 512» и повтори.'
                self.events.put(('error', text))

        threading.Thread(target=worker, daemon=True, name='headswap-inference').start()

    def stop(self):
        self.cancel.set()
        self.cancel_button.configure(state='disabled')
        self.status.set('Останавливаю после текущего этапа…')

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'progress':
                    if not self.cancel.is_set():
                        self.status.set(value)
                else:
                    if kind == 'done':
                        self.result, self.result_seed = value
                        self.view.set('Результат')
                    self.set_busy(False)
                    self.change_view()
                    if kind == 'done':
                        elapsed = round(time.monotonic() - self.started)
                        self.status.set(f'Готово за {elapsed} с. Сравни с исходником и сохрани PNG. Вариант {self.result_seed}.')
                    elif kind == 'cancelled':
                        self.status.set('Обработка остановлена. Можно попробовать снова.')
                    else:
                        self.status.set('Не удалось обработать. Подробности в logs/app.log.')
                        if not self.closing:
                            messagebox.showerror('Ошибка обработки', value)
                    if self.closing:
                        self.root.destroy()
                        return
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def save(self):
        if self.result is None:
            return
        name = f'{self.source_path.stem if self.source_path else "photo"}_headswap.png'
        path = filedialog.asksaveasfilename(title='Сохранить результат', defaultextension='.png',
                                           initialfile=name, filetypes=[('PNG без потери качества', '*.png')])
        if path:
            try:
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text('Software', 'HeadSwap - local AI head replacement')
                metadata.add_text('Seed', str(self.result_seed))
                self.result.save(path, pnginfo=metadata)
                self.status.set(f'Сохранено: {Path(path).name}')
            except Exception as error:
                messagebox.showerror('Не удалось сохранить', str(error))

    def close(self):
        if self.busy:
            self.closing = True
            self.stop()
            self.status.set('Останавливаю обработку и закрываю приложение после текущего этапа…')
        else:
            self.root.destroy()


if __name__ == '__main__':
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    application = Application(root)
    import argparse
    parser = argparse.ArgumentParser(description='Локальная замена головы на фото')
    parser.add_argument('--photo', help='Открыть указанное исходное фото')
    args = parser.parse_args()
    if args.photo:
        application.load_main(args.photo)
    root.mainloop()
