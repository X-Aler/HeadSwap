"""Local FLUX.2 Klein NF4 + BFS head transfer, optimized for 6 GB NVIDIA GPUs."""
import gc
import os
from pathlib import Path
from threading import Event
from imaging import context_box, model_size, composite_head, validate_boxes

ROOT = Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')

PROMPT = (
    'head_swap: image 1 is the base scene. Completely remove the central person\'s '
    'original head, face, hair, hat, helmet and all original headwear from image 1. '
    'Replace them with ONLY the head from image 2: copy its identity, facial '
    'features, skull shape, jawline, ears, hairstyle and headwear. If the person '
    'in image 2 has no hat, the result must have no hat and no remnants of the '
    'original hat. Match the head rotation, gaze, expression and lighting of '
    'image 1. Preserve the body, clothing BELOW THE NECK, body pose, background, '
    'other people and composition of image 1. Natural head proportions, seamless '
    'neck transition. Preserve the visual style of image 1.'
)


class Cancelled(Exception):
    pass


class HeadSwapEngine:
    def __init__(self):
        self.pipe = None
        self.prompt_embeds = None

    def load(self, progress, cancel):
        if self.pipe is not None:
            return
        if not (ROOT / 'models' / 'ready.json').exists():
            raise RuntimeError('Модели ещё не установлены. Закрой приложение и запусти Install.cmd.')
        progress('Загружаю нейросеть в память… Первый запуск дольше обычного.')
        import torch
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        from transformers import Qwen3ForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError('Не найдена NVIDIA CUDA. Установи драйвер NVIDIA и запусти Install.cmd.')
        if cancel.is_set():
            raise Cancelled()
        # Stage each NF4 component separately: loading the complete pipeline on
        # CUDA first would momentarily require more than the 6 GB card can hold.
        progress('Загружаю текстовый модуль…')
        encoder = Qwen3ForCausalLM.from_pretrained(
            str(ROOT / 'models' / 'klein' / 'text_encoder'),
            dtype=torch.float16, device_map={'': 0}, local_files_only=True,
        )
        encoder.to('cpu')
        encoder.hf_device_map = None
        torch.cuda.empty_cache()
        if cancel.is_set():
            del encoder
            raise Cancelled()
        progress('Загружаю модуль замены головы…')
        transformer = Flux2Transformer2DModel.from_pretrained(
            str(ROOT / 'models' / 'klein' / 'transformer'),
            torch_dtype=torch.float16, device_map={'': 'cuda:0'},
            local_files_only=True, low_cpu_mem_usage=True,
        )
        transformer.to('cpu')
        transformer.hf_device_map = None
        torch.cuda.empty_cache()
        pipe = Flux2KleinPipeline.from_pretrained(
            str(ROOT / 'models' / 'klein'), torch_dtype=torch.float16,
            local_files_only=True, low_cpu_mem_usage=True,
            text_encoder=encoder, transformer=transformer,
        )
        pipe.load_lora_weights(str(ROOT / 'models' / 'bfs'),
                               weight_name='bfs_head_v1_flux-klein_4b.safetensors',
                               local_files_only=True)
        # The VAE decoder is small; FP32 avoids overflow artifacts on Turing.
        pipe.vae.to(dtype=torch.float32)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        if cancel.is_set():
            self.unload()
            raise Cancelled()

    def unload(self):
        if self.pipe is not None:
            self.pipe.maybe_free_model_hooks()
            self.pipe = None
        self.prompt_embeds = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def swap(self, photo, boxes, reference, resolution, seed, progress, cancel=None):
        cancel = cancel or Event()
        mutual = reference is None
        validate_boxes(boxes, photo.size, 2 if mutual else 1)
        self.load(progress, cancel)
        import torch
        from PIL import Image

        original = photo.copy()
        result = original.copy()
        targets = list(boxes) if mutual else [boxes[0]]
        # Both donors come from the ORIGINAL, never an already swapped result.
        donors = [original.crop(boxes[1].tuple()), original.crop(boxes[0].tuple())] if mutual else [reference]
        try:
            # The edit instruction is fixed. Encode it once per application,
            # rather than running the 4B text encoder again for every head.
            if self.prompt_embeds is None:
                progress('Подготавливаю инструкцию для нейросети…')
                with torch.inference_mode():
                    embeds, _ = self.pipe.encode_prompt(prompt=PROMPT, device=torch.device('cuda'))
                    if not bool(torch.isfinite(embeds).all()):
                        raise RuntimeError('Численная ошибка текстового модуля. Подробности — в logs/app.log.')
                    self.prompt_embeds = embeds.detach().cpu()
                self.pipe.maybe_free_model_hooks()
            for index, (head, donor) in enumerate(zip(targets, donors)):
                if cancel.is_set():
                    raise Cancelled()
                context = context_box(head, original.size)
                crop = original.crop(context.tuple())
                width, height = model_size(crop.size, resolution)
                target = crop.resize((width, height), Image.Resampling.LANCZOS)
                donor = donor.resize(model_size(donor.size, resolution), Image.Resampling.LANCZOS)
                progress(f'Голова {index + 1} из {len(targets)} · подготовка изображения…')

                def on_step(pipe, step, timestep, values):
                    if cancel.is_set():
                        raise Cancelled()
                    progress(f'Голова {index + 1} из {len(targets)} · шаг {step + 1} из 4')
                    return values

                with torch.inference_mode():
                    generated = self.pipe(
                        image=[target, donor], prompt_embeds=self.prompt_embeds.to('cuda'),
                        height=height, width=width,
                        num_inference_steps=4, guidance_scale=1.0,
                        generator=torch.Generator('cpu').manual_seed(seed + index),
                        callback_on_step_end=on_step,
                    ).images[0]
                # Never allow a numerical failure to be saved as a black success.
                import numpy as np
                pixels = np.asarray(generated)
                if float(pixels.std()) < 0.5:
                    raise RuntimeError('Модель вернула пустое изображение. Попробуй другой вариант или меньшее разрешение.')
                result = composite_head(result, generated, context, head)
            if cancel.is_set():
                raise Cancelled()
            return result
        finally:
            self.pipe.maybe_free_model_hooks()
            gc.collect()
            torch.cuda.empty_cache()
