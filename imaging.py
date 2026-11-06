"""Image coordinates and local compositing; does not import the neural model."""
from dataclasses import dataclass
import math
from statistics import median
from PIL import Image, ImageChops, ImageOps


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def tuple(self):
        return self.x1, self.y1, self.x2, self.y2

    def clamp(self, size):
        w, h = size
        x1, x2 = sorted((max(0, min(w, self.x1)), max(0, min(w, self.x2))))
        y1, y2 = sorted((max(0, min(h, self.y1)), max(0, min(h, self.y2))))
        return Box(x1, y1, x2, y2)

    def overlaps(self, other):
        return (self.x1 < other.x2 and other.x1 < self.x2
                and self.y1 < other.y2 and other.y1 < self.y2)


def open_photo(path):
    with Image.open(path) as image:
        if image.width * image.height > 60_000_000:
            raise ValueError('Фото слишком большое. Максимум — 60 мегапикселей.')
        return ImageOps.exif_transpose(image).convert('RGB')


def context_box(head, size):
    """Give the model neck/shoulders and background to guide pose and lighting."""
    padx, pady = round(head.width * 0.4), round(head.height * 0.3)
    return Box(head.x1 - padx, head.y1 - pady,
               head.x2 + padx, head.y2 + round(head.height * 0.55)).clamp(size)


def model_size(size, limit):
    w, h = size
    scale = min(limit / w, limit / h)
    return max(64, round(w * scale / 16) * 16), max(64, round(h * scale / 16) * 16)


def match_flat_background(original, generated):
    """Restore a flat background's color without recoloring the new head.

    Apply only when all four inset corners of the original have the same color
    and the corresponding generated corners still look like that background.
    Textured photo backgrounds are left alone.
    """
    w, h = original.size
    dx, dy = max(1, w // 10), max(1, h // 10)
    points = [(dx, dy), (w - dx - 1, dy), (dx, h - dy - 1), (w - dx - 1, h - dy - 1)]
    old = [original.getpixel(point) for point in points]
    new = [generated.getpixel(point) for point in points]
    if any(max(c[channel] for c in old) - min(c[channel] for c in old) > 2 for channel in range(3)):
        return generated
    background = tuple(round(median(c[channel] for c in old)) for channel in range(3))
    if any(max(abs(c[channel] - background[channel]) for channel in range(3)) > 32 for c in new):
        return generated
    if any(max(c[channel] for c in new) - min(c[channel] for c in new) > 16 for channel in range(3)):
        return generated
    shifted = tuple(round(median(c[channel] for c in new)) for channel in range(3))

    def similar_color(color):
        channels = ImageChops.difference(generated, Image.new('RGB', generated.size, color)).split()
        distance = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
        return distance.point(lambda value: 255 if value <= 4 else max(0, (8 - value) * 255 // 4))

    mask = ImageChops.lighter(similar_color(background), similar_color(shifted))
    corrected = generated.copy()
    corrected.paste(Image.new('RGB', generated.size, background), (0, 0), mask)
    return corrected


def composite_head(base, generated, context, head, feather=0.06):
    """Outside the head rectangle every pixel is exactly from the original."""
    generated = generated.resize((context.width, context.height), Image.Resampling.LANCZOS)
    relative = (head.x1 - context.x1, head.y1 - context.y1,
                head.x2 - context.x1, head.y2 - context.y1)
    patch = generated.crop(relative)
    patch = match_flat_background(base.crop(head.tuple()), patch)
    radius = max(1, round(min(patch.size) * feather))

    def edge_ramp(length, fade_start, fade_end):
        values = []
        for position in range(length):
            distance = min(position + 0.5 if fade_start else radius,
                           length - position - 0.5 if fade_end else radius)
            fraction = min(1, distance / radius)
            values.append(round(255 * (1 - math.cos(math.pi * fraction)) / 2))
        line = Image.new('L', (length, 1))
        line.putdata(values)
        return line

    # Fully replace the interior; feather only the boundary. At a photo edge
    # there is no surrounding original image to blend, so keep the edit opaque.
    horizontal = edge_ramp(patch.width, head.x1 > 0, head.x2 < base.width).resize(patch.size)
    vertical = edge_ramp(patch.height, head.y1 > 0, head.y2 < base.height).transpose(Image.Transpose.TRANSPOSE).resize(patch.size)
    mask = ImageChops.multiply(horizontal, vertical)
    result = base.copy()
    result.paste(patch, (head.x1, head.y1), mask)
    return result


def validate_boxes(boxes, size, count):
    if len(boxes) != count:
        raise ValueError(f'Выдели голов: {count}. Сейчас выделено: {len(boxes)}.')
    for box in boxes:
        if box != box.clamp(size) or min(box.width, box.height) < 24:
            raise ValueError('Выдели голову целиком, областью не меньше 24 × 24 пикселей.')
    if count == 2 and boxes[0].overlaps(boxes[1]):
        raise ValueError('Области двух голов не должны пересекаться.')
