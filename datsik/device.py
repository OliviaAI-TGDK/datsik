from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map
import torch, os

def load_model(BASE, HF_TOKEN, outdir, gpu_idx=0):
    offload_dir = os.path.join(outdir, "offload")
    os.makedirs(offload_dir, exist_ok=True)

    HAS_CUDA = torch.cuda.is_available()

    if HAS_CUDA:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        try:
            # first try full auto placement
            model = AutoModelForCausalLM.from_pretrained(
                BASE,
                quantization_config=bnb_cfg,
                device_map="auto",
                max_memory={
                    gpu_idx: os.environ.get("CUDA0_MAX_MEM", "6GiB"),
                    "cpu": os.environ.get("CPU_MAX_MEM", "48GiB"),
                },
                offload_folder=offload_dir,
                offload_state_dict=True,
                torch_dtype=torch.float16,
                token=HF_TOKEN,
            )
        except ValueError as e:
            print("[WARN] Auto placement failed, retrying with inferred device_map:", e)

            # lightweight init for inferring device map
            with init_empty_weights():
                model_tmp = AutoModelForCausalLM.from_config(BASE)

            device_map = infer_auto_device_map(
                model_tmp,
                max_memory={
                    gpu_idx: os.environ.get("CUDA0_MAX_MEM", "6GiB"),
                    "cpu": os.environ.get("CPU_MAX_MEM", "48GiB"),
                },
                no_split_module_classes=["MistralDecoderLayer"],
            )

            model = AutoModelForCausalLM.from_pretrained(
                BASE,
                quantization_config=bnb_cfg,
                device_map=device_map,
                offload_folder=offload_dir,
                offload_state_dict=True,
                torch_dtype=torch.float16,
                token=HF_TOKEN,
            )

        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    else:
        # CPU-only fallback
        model = AutoModelForCausalLM.from_pretrained(
            BASE,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
        )
        model.config.use_cache = False

    return model
