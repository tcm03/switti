import torch
from torchvision.transforms import ToPILImage
from PIL.Image import Image as PILImage

from models.vqvae import VQVAEHF
from models.clip import FrozenCLIPEmbedder
from models.switti import SwittiHF, get_crop_condition
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng
import logging

TRAIN_IMAGE_SIZE = (512, 512)

class SwittiPipeline:
    vae_path = "yresearch/VQVAE-Switti"
    text_encoder_path = "openai/clip-vit-large-patch14"
    text_encoder_2_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

    def __init__(self, switti, vae, text_encoder, text_encoder_2,
                 device, dtype=torch.float32,
                 ):
        self.switti = switti.to(dtype)
        self.vae = vae.to(dtype)
        self.text_encoder = text_encoder.to(dtype)
        self.text_encoder_2 = text_encoder_2.to(dtype)

        self.switti.eval()
        self.vae.eval()

        self.device = device

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        torch_dtype=torch.bfloat16,
                        device="cuda",
                        reso=1024,
                        ):
        switti = SwittiHF.from_pretrained(pretrained_model_name_or_path).to(device)
        vae = VQVAEHF.from_pretrained(cls.vae_path, reso=reso).to(device)
        text_encoder = FrozenCLIPEmbedder(cls.text_encoder_path, device=device)
        text_encoder_2 = FrozenCLIPEmbedder(cls.text_encoder_2_path, device=device)

        return cls(switti, vae, text_encoder, text_encoder_2, device, torch_dtype)

    @staticmethod
    def to_image(tensor):
        return [ToPILImage()(
            (255 * img.cpu().detach()).to(torch.uint8))
        for img in tensor]

    def _encode_prompt(self, prompt: str | list[str]):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        encodings = [
            self.text_encoder.encode(prompt),
            self.text_encoder_2.encode(prompt),
        ]
        logging.info(f"@tcm In SwittiPipeline._encode_prompt(): {type(self.text_encoder)} embedding shape = {encodings[0].last_hidden_state.shape}")
        # [2025-09-03] @tcm: text_encoder embedding shape = [B, 77, 768]
        logging.info(f"@tcm In SwittiPipeline._encode_prompt(): {type(self.text_encoder_2)} embedding shape = {encodings[1].last_hidden_state.shape}")
        # [2025-09-03] @tcm: text_encoder_2 embedding shape = [B, 77, 1280]
        prompt_embeds = torch.concat(
            [encoding.last_hidden_state for encoding in encodings], dim=-1
        )
        logging.info(f"@tcm In SwittiPipeline._encode_prompt(): prompt_embeds shape = {prompt_embeds.shape}")
        # [2025-09-03] @tcm: prompt_embeds shape = [B, 77, 2048]
        pooled_prompt_embeds = encodings[-1].pooler_output
        logging.info(f"@tcm In SwittiPipeline._encode_prompt(): pooled_prompt_embeds shape = {pooled_prompt_embeds.shape}")
        # [2025-09-03] @tcm: pooled_prompt_embeds shape = [B, 1280]
        attn_bias = encodings[-1].attn_bias
        logging.info(f"@tcm In SwittiPipeline._encode_prompt(): attn_bias shape = {attn_bias.shape}")
        # [2025-09-03] @tcm: attn_bias shape = [B, 77]

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    def encode_prompt(
        self,
        prompt: str | list[str],
        null_prompt: str = "",
        encode_null: bool = True,
    ):
        prompt_embeds, pooled_prompt_embeds, attn_bias = self._encode_prompt(prompt)
        # [2025-09-03] @tcm: prompt_embeds shape = [B, 77, 768+1280]
        # [2025-09-03] @tcm: pooled_prompt_embeds shape = [B, 1280]
        # [2025-09-03] @tcm: attn_bias shape = [B, 77]
        if encode_null:
            logging.info(f"@tcm In SwittiPipeline.__call__(): ENCODING NULL_PROMPT...")
            B, L, hidden_dim = prompt_embeds.shape
            pooled_dim = pooled_prompt_embeds.shape[1]

            null_embeds, null_pooled_embeds, null_attn_bias = self._encode_prompt(null_prompt)
            
            null_embeds = null_embeds[:, :L].expand(B, L, hidden_dim).to(prompt_embeds.device)
            logging.info(f"@tcm In SwittiPipeline.__call__(): null_embeds shape = {null_embeds.shape}")
            # [2025-09-03] @tcm: null_embeds shape = [B, 77, 768+1280]
            null_pooled_embeds = null_pooled_embeds.expand(B, pooled_dim).to(pooled_prompt_embeds.device)
            logging.info(f"@tcm In SwittiPipeline.__call__(): null_pooled_embeds shape = {null_pooled_embeds.shape}")
            # [2025-09-03] @tcm: null_pooled_embeds shape = [B, 1280]
            null_attn_bias = null_attn_bias[:, :L].expand(B, L).to(attn_bias.device)
            logging.info(f"@tcm In SwittiPipeline.__call__(): null_attn_bias shape = {null_attn_bias.shape}")
            # [2025-09-03] @tcm: null_attn_bias shape = [B, 77]

            prompt_embeds = torch.cat([prompt_embeds, null_embeds], dim=0)
            logging.info(f"@tcm In SwittiPipeline.__call__(): prompt_embeds shape = {prompt_embeds.shape}")
            # [2025-09-03] @tcm: prompt_embeds shape = [2*B, 77, 768+1280]
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, null_pooled_embeds], dim=0)
            logging.info(f"@tcm In SwittiPipeline.__call__(): pooled_prompt_embeds shape = {pooled_prompt_embeds.shape}")
            # [2025-09-03] @tcm: pooled_prompt_embeds shape = [2*B, 1280]
            attn_bias = torch.cat([attn_bias, null_attn_bias], dim=0)
            logging.info(f"@tcm In SwittiPipeline.__call__(): attn_bias shape = {attn_bias.shape}")
            # [2025-09-03] @tcm: attn_bias shape = [2*B, 77]

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | list[str],
        null_prompt: str = "",
        seed: int | None = None,
        cfg: float = 6.,
        top_k: int = 400,
        top_p: float = 0.95,
        more_smooth: bool = False,
        return_pil: bool = True,
        smooth_start_si: int = 0,
        turn_off_cfg_start_si: int = 10,
        turn_on_cfg_start_si: int = 0,
        last_scale_temp: None | float = None,
    ) -> torch.Tensor | list[PILImage]:
        """
        only used for inference, on autoregressive mode
        :param prompt: text prompt to generate an image
        :param null_prompt: negative prompt for CFG
        :param seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: sampling using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if return_pil: list of PIL Images, else: torch.tensor (B, 3, H, W) in [0, 1]
        """
        assert not self.switti.training
        switti = self.switti
        vae = self.vae
        vae_quant = self.vae.quantize
        if seed is None:
            rng = None
        else:
            switti.rng.manual_seed(seed)
            rng = switti.rng

        logging.info(f"@tcm In SwittiPipeline.__call__(): prompt = {prompt}")
        logging.info(f"@tcm In SwittiPipeline.__call__(): null_prompt = {null_prompt}")
        # [2025-09-03] @tcm: prompt is List[str] of instructions for image generation
        context, cond_vector, context_attn_bias = self.encode_prompt(prompt, null_prompt)
        logging.info(f"@tcm In SwittiPipeline.__call__(): context.shape = {context.shape}")
        # [2025-09-03] @tcm: context.shape = [2*B, 77, 2048]
        logging.info(f"@tcm In SwittiPipeline.__call__(): cond_vector.shape = {cond_vector.shape}")
        # [2025-09-03] @tcm: cond_vector.shape = [2*B, 1280]
        logging.info(f"@tcm In SwittiPipeline.__call__(): context_attn_bias.shape = {context_attn_bias.shape}")
        # [2025-09-03] @tcm: context_attn_bias.shape = [2*B, 77]
        B = context.shape[0] // 2

        cond_vector = switti.text_pooler(cond_vector) # [2025-09-03] @tcm: text_pooler = nn.Linear(pooled_embed_size=1280, D=1920)
        logging.info(f"@tcm In SwittiPipeline.__call__(): after text_pooler: cond_vector.shape = {cond_vector.shape}")
        # [2025-09-03] @tcm: after text_pooler: cond_vector.shape = [2*B, 1920]
        if switti.use_crop_cond:
            crop_coords = get_crop_condition(2 * B * [TRAIN_IMAGE_SIZE[0]],
                                             2 * B * [TRAIN_IMAGE_SIZE[1]],
                                             ).to(cond_vector.device)
            logging.info(f"@tcm In SwittiPipeline.__call__(): crop_coords = {crop_coords}")
            # [2025-09-03] @tcm: crop_coords = tensor([[512, 512, 0, 0] x 10])
            crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, switti.D)
            logging.info(f"@tcm In SwittiPipeline.__call__(): crop_embed.shape = {crop_embed.shape}")
            # [2025-09-03] @tcm: crop_embed.shape = [2*B, 1920]
            crop_cond = switti.crop_proj(crop_embed)
            logging.info(f"@tcm In SwittiPipeline.__call__(): crop_cond.shape = {crop_cond.shape}")
            # [2025-09-03] @tcm: crop_cond.shape = [2*B, 1920]
        else:
            crop_cond = None
        logging.info(f"@tcm In SwittiPipeline.__call__(): crop_cond.shape = {crop_cond.shape if crop_cond is not None else None}")
        # [2025-09-03] @tcm: crop_cond.shape = [2*B, 1920]
        sos = cond_BD = cond_vector

        lvl_pos = switti.lvl_embed(switti.lvl_1L)
        if not switti.rope:
            lvl_pos += switti.pos_1LC
        next_token_map = (
            sos.unsqueeze(1)
            + switti.pos_start.expand(2 * B, switti.first_l, -1)
            + lvl_pos[:, : switti.first_l]
        )
        logging.info(f"@tcm In SwittiPipeline.__call__(): next_token_map.shape = {next_token_map.shape}")
        # [2025-09-03] @tcm: next_token_map.shape = [2*B, 1, 1920]
        cur_L = 0
        f_hat = sos.new_zeros(B, switti.Cvae, switti.patch_nums[-1], switti.patch_nums[-1])
        logging.info(f"@tcm In SwittiPipeline.__call__(): before forward through self.blocks: f_hat.shape = {f_hat.shape}")
        # [2025-09-03] @tcm: f_hat.shape = [B, 32, 64, 64]

        for b in switti.blocks:
            b.attn.kv_caching(switti.use_ar) # Use KV caching if switti is in the AR mode 
            b.cross_attn.kv_caching(True)

        # [2025-09-03] @tcm: switti.patch_nums[14] = [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64]
        for si, pn in enumerate(switti.patch_nums):  # si: i-th segment
            ratio = si / switti.num_stages_minus_1
            logging.info(f"@tcm In SwittiPipeline.__call__(): segment {si}, patch num = {pn}, ratio = {ratio}")
            x_BLC = next_token_map

            if switti.rope:
                freqs_cis = switti.freqs_cis[:, cur_L : cur_L + pn * pn]
            else:
                freqs_cis = switti.freqs_cis
            logging.info(f"@tcm In SwittiPipeline.__call__(): freqs_cis.shape = {freqs_cis.shape}")
            # [2025-09-03] @tcm: freqs_cis.shape = [1, s^2, 32], s \in switti.patch_nums

            if si >= turn_off_cfg_start_si:
                # [2025-09-03] @tcm: disable cfg at high resolutions (from scale 27x27)
                apply_smooth = False
                x_BLC = x_BLC[:B]
                context = context[:B]
                context_attn_bias = context_attn_bias[:B]
                freqs_cis = freqs_cis[:B]
                cond_BD = cond_BD[:B]
                if crop_cond is not None:
                    crop_cond = crop_cond[:B]
                for b in switti.blocks:
                    if b.attn.caching and b.attn.cached_k is not None:
                        b.attn.cached_k = b.attn.cached_k[:B]
                        b.attn.cached_v = b.attn.cached_v[:B]
                    if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                        b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                        b.cross_attn.cached_v = b.cross_attn.cached_v[:B]
            else:
                apply_smooth = more_smooth

            for block in switti.blocks:
                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=None,
                    context=context,
                    context_attn_bias=context_attn_bias,
                    freqs_cis=freqs_cis,
                    crop_cond=crop_cond,
                )
            cur_L += pn * pn
            logging.info(f"@tcm In SwittiPipeline.__call__(): after forward through switti.blocks, x_BLC.shape = {x_BLC.shape}")
            # [2025-09-03] @tcm: x_BLC.shape = [B, s^2, 1920], s \in switti.patch_nums
            logging.info(f"@tcm In SwittiPipeline.__call__(): after forward through switti.blocks, cur_L = {cur_L}")
            # [2025-09-03] @tcm: cur_L = 5355

            logits_BlV = switti.get_logits(x_BLC, cond_BD)
            logging.info(f"@tcm In SwittiPipeline.__call__(): logits_BlV.shape = {logits_BlV.shape}")
            # [2025-09-03] @tcm: logits_BlV.shape = [B, s^2, 4096], s \in switti.patch_nums
            # Guidance
            if si < turn_on_cfg_start_si:
                logits_BlV = logits_BlV[:B]
            elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            elif last_scale_temp is not None:
                logits_BlV = logits_BlV / last_scale_temp

            if apply_smooth and si >= smooth_start_si:
                # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                idx_Bl = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng,
                )
                h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
                logging.info(f"@tcm In SwittiPipeline.__call__(): apply_smooth, idx_Bl.shape = {idx_Bl.shape}, h_BChw.shape = {h_BChw.shape}")
            else:
                # default nucleus sampling
                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1,
                )[:, :, 0]
                h_BChw = vae_quant.embedding(idx_Bl)
                logging.info(f"@tcm In SwittiPipeline.__call__(): default nucleus sampling, idx_Bl.shape = {idx_Bl.shape}, h_BChw.shape = {h_BChw.shape}")
                # [2025-09-03] @tcm: idx_Bl.shape = [B, s^2], s \in switti.patch_nums
                # [2025-09-03] @tcm: h_BChw.shape = [B, s^2, 32], s \in switti.patch_nums
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, switti.Cvae, pn, pn)
            logging.info(f"@tcm In SwittiPipeline.__call__(): transposed h_BChw.shape = {h_BChw.shape}")
            # [2025-09-03] @tcm: transposed h_BChw.shape = [B, 32, s, s], s \in switti.patch_nums
            f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
                    si, len(switti.patch_nums), f_hat, h_BChw,
            )
            logging.info(f"@tcm In SwittiPipeline.__call__(): after get_next_autoregressive_input, f_hat.shape = {f_hat.shape}, next_token_map.shape = {next_token_map.shape}")
            # [2025-09-03] @tcm: f_hat.shape = [B, 32, 64, 64]
            # [2025-09-03] @tcm: next_token_map.shape = [B, 32, s_{i+1}, s_{i+1}]
            if si != switti.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    switti.word_embed(next_token_map)
                    + lvl_pos[:, cur_L : cur_L + switti.patch_nums[si + 1] ** 2]
                )
                # double the batch sizes due to CFG
                next_token_map = next_token_map.repeat(2, 1, 1)

        for b in switti.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)

        # de-normalize, from [-1, 1] to [0, 1]
        img = vae.fhat_to_img(f_hat).add(1).mul(0.5)
        if return_pil:
            img = self.to_image(img)

        return img
