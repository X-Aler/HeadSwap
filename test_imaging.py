import unittest
from PIL import Image, ImageDraw
from imaging import Box, composite_head, context_box, model_size, validate_boxes, match_flat_background


class LocalEditingTests(unittest.TestCase):
    def test_background_is_pixel_identical_outside_edit(self):
        base = Image.effect_noise((200, 160), 70).convert('RGB')
        head = Box(40, 30, 110, 110)
        context = context_box(head, base.size)
        edited = composite_head(base, Image.new('RGB', (128, 128), 'red'), context, head)
        self.assertEqual(edited.size, base.size)
        self.assertNotEqual(edited.getpixel((70, 70)), base.getpixel((70, 70)))
        for y in range(base.height):
            for x in range(base.width):
                if not (head.x1 <= x < head.x2 and head.y1 <= y < head.y2):
                    self.assertEqual(edited.getpixel((x, y)), base.getpixel((x, y)))

    def test_second_swap_preserves_first_and_original_background(self):
        base = Image.new('RGB', (300, 160), 'green')
        a, b = Box(15, 20, 90, 120), Box(180, 20, 260, 120)
        first = composite_head(base, Image.new('RGB', (80, 80), 'red'), context_box(a, base.size), a)
        final = composite_head(first, Image.new('RGB', (80, 80), 'blue'), context_box(b, base.size), b)
        self.assertEqual(final.crop(a.tuple()).tobytes(), first.crop(a.tuple()).tobytes())
        self.assertEqual(final.getpixel((140, 70)), base.getpixel((140, 70)))

    def test_head_touching_photo_edge_does_not_retain_old_head(self):
        base = Image.new('RGB', (100, 100), 'blue')
        head = Box(20, 0, 80, 80)
        edited = composite_head(base, Image.new('RGB', (128, 128), 'red'),
                                context_box(head, base.size), head)
        self.assertEqual(edited.getpixel((50, 0)), (255, 0, 0))
        self.assertEqual(edited.getpixel((50, 10)), (255, 0, 0))
        self.assertEqual(edited.getpixel((5, 5)), (0, 0, 255))

    def test_flat_background_cast_is_removed_without_erasing_new_head(self):
        original = Image.new('RGB', (100, 120), (197, 219, 220))
        ImageDraw.Draw(original).ellipse((25, 20, 75, 100), fill=(235, 195, 162))
        generated = Image.new('RGB', original.size, (185, 207, 209))
        ImageDraw.Draw(generated).rectangle((20, 25, 80, 95), fill=(181, 215, 244))
        corrected = match_flat_background(original, generated)
        self.assertEqual(corrected.getpixel((10, 30)), (197, 219, 220))
        self.assertEqual(corrected.getpixel((50, 50)), (181, 215, 244))

    def test_reject_overlapping_or_tiny_selections(self):
        with self.assertRaises(ValueError):
            validate_boxes([Box(0, 0, 50, 50), Box(40, 40, 90, 90)], (100, 100), 2)
        with self.assertRaises(ValueError):
            validate_boxes([Box(0, 0, 10, 10)], (100, 100), 1)
        with self.assertRaises(ValueError):
            validate_boxes([Box(-10, 0, 50, 50)], (100, 100), 1)

    def test_inference_dimensions_are_bounded_and_aligned(self):
        for size in [(200, 500), (1000, 300), (300, 300), (50, 5000)]:
            w, h = model_size(size, 512)
            self.assertEqual(w % 16, 0)
            self.assertEqual(h % 16, 0)
            self.assertLessEqual(max(w, h), 512)
            self.assertGreaterEqual(min(w, h), 64)


if __name__ == '__main__':
    unittest.main()
