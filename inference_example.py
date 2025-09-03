import torch
from models import SwittiPipeline
from torchvision.utils import make_grid
from calculate_metrics import to_PIL_image

import logging
# Configure logging with line numbers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)

def main():
    device = 'cuda:0'

    model_path = "yresearch/Switti-1024"
    pipe = SwittiPipeline.from_pretrained(model_path, device=device, torch_dtype=torch.bfloat16)



    prompts = ["a rainbow bunny holds a board with the text 'diffusion is dead'",
            "A close-up photograph of a Corgi dog. The dog is wearing a black hat and round, dark sunglasses. The Corgi has a joyful expression, with its mouth open and tongue sticking out, giving an impression of happiness or excitement",
            "a green sphere on the top of a yellow sphere, the behind is a red triangle. A cat is on the right and a dog is on the left",
            "Create a mesmerizing image of three intricately designed potions displayed on an ornate, antique wooden table within a charming old apothecary. The first potion is a captivating cobalt orange, housed in a stunning pentagon-shaped glass bottle that sparkles with its many facets; its label, meticulously crafted with delicate silver filigree and botanical illustrations of ethereal flowers, prominently features the letters “I” in an ornate, swirling script, while a silver ribbon interwoven with tiny sapphire beads wraps around the neck, adorned with a charm in the shape of a crescent moon. The second potion is a rich crimson red, contained in a flat, oval-shaped glass bottle adorned with intricate engravings of mystical symbols, including runes and ancient scripts; its label displays the letters “N” in embossed gold leaf, framed by elaborate floral designs, and is topped with a cork stopper embellished with a miniature brass key and tiny ruby gemstones. The third potion is a vivid emerald green, held in a sleek square glass bottle featuring enchanting etchings of mythical creatures like dragons and phoenixes; its scroll-like label, crafted from aged parchment, prominently features the letter “F” intertwined with ancient alchemical symbols and delicate vine patterns. All three bottles are approximately the same height, creating a harmonious display against a backdrop filled with shelves overflowing with dried herbs, colorful glass jars, and ancient scrolls, all illuminated by soft, warm light filtering through a stained-glass window, enhancing the magical atmosphere of the apothecary",
            "Create an image with the text 'Stay Negative' in an uplifting style, featuring bright and cheerful colors, swirling floral patterns, and a radiant sun in the background"
            ]
    images = pipe(prompts,
                cfg=6.0,
                top_k=400,
                top_p=0.95,
                more_smooth=True,
                return_pil=False,
                smooth_start_si=2,
                turn_on_cfg_start_si=2,
                turn_off_cfg_start_si=11,
                last_scale_temp=0.1,
                seed=59,
                )
    logging.info(f"@tcm In main(): type(images) = {type(images)}")
    pil_images = to_PIL_image(make_grid([img.float() for img in images], nrow=2))
    logging.info(f"@tcm In main(): type(pil_images) = {type(pil_images)}")
    pil_images.save("inference_example.png")

if __name__ == "__main__":
    main()