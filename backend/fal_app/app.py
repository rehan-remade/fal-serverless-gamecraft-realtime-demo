import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image

import fal
from fal.container import ContainerImage
from fal.toolkit import clone_repository, download_file, Video, FAL_PERSISTENT_DIR
import requests

from fal_app.schema import GenerateInput, GenerateOutput

# Docker container definition
dockerfile_str = r"""
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Cache bust: v9
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils python3-pip python3.11-dev \
    git ffmpeg curl ca-certificates build-essential && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Torch/cu124 with correct versions for compatibility
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Install all GameCraft requirements
RUN pip install \
    accelerate==1.9.0 \
    av==15.0.0 \
    certifi==2025.8.3 \
    charset-normalizer==3.4.2 \
    contourpy==1.3.2 \
    cycler==0.12.1 \
    decord==0.6.0 \
    diffusers==0.34.0 \
    einops==0.8.1 \
    filelock==3.13.1 \
    fonttools==4.59.0 \
    fsspec==2024.6.1 \
    hf-xet==1.1.5 \
    huggingface-hub==0.34.3 \
    idna==3.10 \
    imageio==2.37.0 \
    imageio-ffmpeg==0.6.0 \
    importlib_metadata==8.7.0 \
    Jinja2==3.1.4 \
    kiwisolver==1.4.8 \
    loguru==0.7.3 \
    MarkupSafe==2.1.5 \
    matplotlib==3.10.5 \
    mpmath==1.3.0 \
    networkx==3.3 \
    ninja==1.11.1.4 \
    numpy==2.1.2 \
    nvidia-ml-py==12.575.51 \
    nvidia-nccl-cu12==2.21.5 \
    nvidia-nvjitlink-cu12==12.4.127 \
    nvidia-nvtx-cu12==12.4.127 \
    nvitop==1.5.2 \
    opencv-python-headless==4.12.0.88 \
    packaging==25.0 \
    pandas==2.3.1 \
    pillow==11.0.0 \
    protobuf==6.31.1 \
    psutil==7.0.0 \
    pyparsing==3.2.3 \
    python-dateutil==2.9.0.post0 \
    pytz==2025.2 \
    PyYAML==6.0.2 \
    regex==2025.7.34 \
    requests==2.32.4 \
    safetensors==0.5.3 \
    sentencepiece==0.2.0 \
    setuptools==78.1.1 \
    six==1.17.0 \
    sympy==1.13.1 \
    tokenizers==0.21.4 \
    tqdm==4.67.1 \
    transformers==4.54.1 \
    triton==3.1.0 \
    typing_extensions==4.12.2 \
    tzdata==2025.2 \
    urllib3==2.5.0 \
    wheel==0.45.1 \
    zipp==3.23.0

# Install ninja first (required for flash attention)
RUN pip install ninja

# Install flash attention v2
RUN pip install flash-attn --no-build-isolation

"""

CustomImage = ContainerImage.from_dockerfile_str(dockerfile_str)


class GameCraft(
    fal.App,
    name="gamecraft",
    kind="container",
    image=CustomImage,
    min_concurrency=0,
    max_concurrency=1,
    keep_alive=300,
):  # type: ignore
    machine_type = "GPU-H100"
    num_gpus = 1
    local_python_modules = ["fal_app"]
    requirements = [
        # Core ML packages
        "torch==2.5.1",
        "torchvision==0.20.1", 
        "transformers==4.54.1",
        "diffusers==0.34.0",
        "accelerate==1.9.0",
        
        # Hugging Face
        "huggingface-hub==0.34.3",
        "hf-xet==1.1.5",
        "tokenizers==0.21.4",
        "safetensors==0.5.3",
        
        
        # Video/Image processing
        "av==15.0.0",
        "decord==0.6.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "opencv-python-headless==4.12.0.88",
        "pillow==11.0.0",
        
        # Scientific computing
        "numpy==2.1.2",
        "pandas==2.3.1",
        "matplotlib==3.10.5",
        "contourpy==1.3.3",
        "einops==0.8.1",
        
        # Utilities
        "loguru==0.7.3",
        "tqdm==4.67.1",
        "pyyaml==6.0.2",
        "packaging==25.0",
        "filelock==3.13.1",
        "fsspec==2024.6.1",
        
        # Other dependencies
        "requests==2.32.4",
        "jinja2==3.1.4",
        "markupsafe==2.1.5",
        "sentencepiece==0.2.0",
        "regex==2025.7.34",
        "psutil==7.0.0",
        "sympy==1.13.1",
        "mpmath==1.3.0",
        "networkx==3.3",
        
        # Date/time
        "pytz==2025.2",
        "python-dateutil==2.9.0.post0",
        "tzdata==2025.2",
        
        # Parsing/formatting
        "pyparsing==3.2.3",
        "cycler==0.12.1",
        "kiwisolver==1.4.8",
        "fonttools==4.59.0",
        
        # Additional
        "certifi==2025.8.3",
        "charset-normalizer==3.4.2",
        "idna==3.10",
        "urllib3==2.5.0",
        "six==1.17.0",
        "typing_extensions==4.12.2",
        "importlib_metadata==8.7.0",
        "zipp==3.23.0",
        
        # NVIDIA specific (may not need all of these in requirements)
        "triton==3.1.0"
    ]

    def setup(self):
        """
        Set up models with support for both single and multi-GPU warming
        """
        from huggingface_hub import hf_hub_download, snapshot_download
        from loguru import logger
        
        print("=== Starting GameCraft Setup ===")
        
        # Install flash-attn with all build dependencies
        print("📦 Ensuring flash-attn is available...")
        import subprocess
        import sys
        try:
            import flash_attn
            print("✅ flash-attn already available")
        except ImportError:
            print("Installing build dependencies and flash-attn...")
            try:
                # Install all build dependencies
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "setuptools", "wheel", "ninja", "packaging"
                ])
                print("✅ Build dependencies installed")
                
                # Make sure torch is available for the build
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu124"
                ])
                print("✅ PyTorch installed")
                
                # Then install flash-attn
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "flash-attn", "--no-build-isolation"
                ])
                print("✅ flash-attn installed")
                
                # Import it to verify
                import flash_attn
                print("✅ flash-attn imported successfully")
                
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to install flash-attn: {e}")
                print("Continuing without flash-attn - may fall back to standard attention")
            except ImportError as e:
                print(f"⚠️ flash-attn installed but can't import: {e}")
                print("Continuing without flash-attn")
        
        # 🚀 Clone your forked repository
        print("📦 Cloning GameCraft repository...")
        self.repo_path = Path(clone_repository(
            "https://github.com/Pheat-AI/Hunyuan-GameCraft-1.0",
            target_dir="/tmp",
            include_to_path=True,
            repo_name="hunyuan-gamecraft",
        ))
        
        print(f"✅ Repository cloned to: {self.repo_path}")
        
        # Now we can import from the cloned repository
        from hymm_sp.warm import model_warmer
        from hymm_sp.sample import generate_video
        
        # Store references for use in generate endpoint
        self.model_warmer = model_warmer
        self.generate_video = generate_video
        
        # Use /data for persistent model storage
        self.models_base_dir = Path("/data/models/hunyuan_gamecraft")
        self.models_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up model directories in /data
        self.gamecraft_models_dir = self.models_base_dir / "gamecraft_models"
        
        # Create all model directories
        self.gamecraft_models_dir.mkdir(parents=True, exist_ok=True)
        
        # VAE directory
        self.stdmodels_dir = self.models_base_dir / "stdmodels"
        self.vae_dir = self.stdmodels_dir / "vae_3d" / "hyvae"
        self.vae_dir.mkdir(parents=True, exist_ok=True)
        
        # CLIP directory
        self.clip_dir = self.stdmodels_dir / "openai_clip-vit-large-patch14"
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        
        # LLaVA directory
        self.llava_dir = self.stdmodels_dir / "llava-llama-3-8b-v1_1-transformers"
        self.llava_dir.mkdir(parents=True, exist_ok=True)

        # Enable HF transfer for faster downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["HF_HOME"] = "/data/.cache/huggingface"
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "1000000000000"
        os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "32"
        
        # Set up environment variables for model loading
        os.environ["MODEL_BASE"] = str(self.stdmodels_dir)
        
        try:
            print("=== Downloading GameCraft model weights ===")
            
            # Check if distilled checkpoint already exists
            distilled_checkpoint_path = self.gamecraft_models_dir / "mp_rank_00_model_states_distill.pt"
            
            if not distilled_checkpoint_path.exists():
                print("Downloading distilled checkpoint...")
                downloaded_path = hf_hub_download(
                    repo_id="tencent/Hunyuan-GameCraft-1.0",
                    filename="gamecraft_models/mp_rank_00_model_states_distill.pt",
                    cache_dir="/data/.cache/huggingface",
                    local_dir_use_symlinks=False
                )
                
                # Copy to expected location
                shutil.copy2(downloaded_path, distilled_checkpoint_path)
                print(f"✓ Distilled checkpoint saved to: {distilled_checkpoint_path}")
            else:
                print(f"✓ Distilled checkpoint already exists at: {distilled_checkpoint_path}")

            # Download VAE model files if not present
            print("\nChecking VAE model files...")
            vae_files = ["pytorch_model.pt", "config.json"]
            for file_name in vae_files:
                target_vae = self.vae_dir / file_name
                if not target_vae.exists():
                    print(f"Downloading VAE file: {file_name}")
                    vae_file_path = hf_hub_download(
                        repo_id="tencent/Hunyuan-GameCraft-1.0",
                        filename=f"stdmodels/vae_3d/hyvae/{file_name}",
                        cache_dir="/data/.cache/huggingface",
                        local_dir_use_symlinks=False
                    )
                    shutil.copy2(vae_file_path, target_vae)
                    print(f"✓ VAE file saved: {target_vae}")
                else:
                    # Verify the file is not empty or corrupted
                    file_size = target_vae.stat().st_size
                    print(f"✓ VAE file already exists: {target_vae} (size: {file_size:,} bytes)")
                    
                    # If pytorch_model.pt is too small, it might be corrupted
                    if file_name == "pytorch_model.pt" and file_size < 1000000:  # Less than 1MB is suspicious
                        print(f"⚠️ VAE model file seems too small ({file_size} bytes), re-downloading...")
                        target_vae.unlink()  # Delete the corrupted file
                        vae_file_path = hf_hub_download(
                            repo_id="tencent/Hunyuan-GameCraft-1.0",
                            filename=f"stdmodels/vae_3d/hyvae/{file_name}",
                            cache_dir="/data/.cache/huggingface",
                            local_dir_use_symlinks=False,
                            force_download=True  # Force re-download
                        )
                        shutil.copy2(vae_file_path, target_vae)
                        new_size = target_vae.stat().st_size
                        print(f"✓ Re-downloaded VAE file: {target_vae} (new size: {new_size:,} bytes)")

            # Download CLIP model if not present
            print("\nChecking CLIP model...")
            clip_config = self.clip_dir / "config.json"
            if not clip_config.exists():
                print("Downloading CLIP model...")
                snapshot_download(
                    repo_id="openai/clip-vit-large-patch14",
                    local_dir=str(self.clip_dir),
                    cache_dir="/data/.cache/huggingface",
                    local_dir_use_symlinks=False
                )
                print(f"✓ CLIP model saved to: {self.clip_dir}")
            else:
                print(f"✓ CLIP model already exists at: {self.clip_dir}")

            # Download LLaVA model if not present
            print("\nChecking LLaVA model...")
            llava_config = self.llava_dir / "config.json"
            if not llava_config.exists():
                print("Downloading LLaVA model...")
                snapshot_download(
                    repo_id="xtuner/llava-llama-3-8b-v1_1-transformers",
                    local_dir=str(self.llava_dir),
                    cache_dir="/data/.cache/huggingface",
                    local_dir_use_symlinks=False
                )
                print(f"✓ LLaVA model saved to: {self.llava_dir}")
            else:
                print(f"✓ LLaVA model already exists at: {self.llava_dir}")

            print("\n=== All models downloaded! ===")
            
            # Set checkpoint path
            self.checkpoint_path = distilled_checkpoint_path
            
            # Verify checkpoint exists
            if not self.checkpoint_path.exists():
                raise RuntimeError(f"Checkpoint not found at {self.checkpoint_path}")
            
            print(f"✓ Using checkpoint: {self.checkpoint_path}")

        except Exception as e:
            print(f"Error downloading models: {e}")
            raise RuntimeError(f"Failed to download required model files: {e}")

        # 🔥 WARM UP MODELS using the cloned repository
        print("\n=== Warming up models ===")
        # Make sure MODEL_BASE is set for the warm function
        os.environ["MODEL_BASE"] = str(self.stdmodels_dir)
        print(f"MODEL_BASE environment variable: {os.environ.get('MODEL_BASE')}")
        print(f"VAE directory exists: {self.vae_dir.exists()}")
        if self.vae_dir.exists():
            print(f"VAE directory contents: {list(self.vae_dir.iterdir())}")

        # Determine number of GPUs to use
        num_gpus = self.num_gpus  # This is 1 in your current config
        
        if num_gpus == 1:
            # Single GPU warming - set env vars for distributed init
            logger.info("🔥 Single GPU warming mode")
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_RANK"] = "0"
            
            # Use existing warming approach
            device = torch.device("cuda")
            self.model_warmer.warm_models(
                checkpoint_path=self.checkpoint_path,
                device=device,
                cpu_offload=False
            )
            
        else:
            # Multi-GPU warming - use torchrun
            logger.info(f"🔥 Multi-GPU warming mode with {num_gpus} GPUs")
            
            # Create warming script in the cloned repo
            warm_script_path = self.repo_path / "hymm_sp/warm_multi_gpu.py"
            
            # Set environment for the subprocess
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.repo_path)
            env["MODEL_BASE"] = str(self.stdmodels_dir)
            env["CHECKPOINT_PATH"] = str(self.checkpoint_path)
            env["USE_FP8"] = "false"  # or "true" based on your needs
            
            # Run torchrun for multi-GPU warming
            cmd = [
                "torchrun",
                "--nnodes=1",
                f"--nproc_per_node={num_gpus}",
                "--master_port", "29605",
                str(warm_script_path)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Multi-GPU warming failed: {result.stderr}")
                raise RuntimeError("Failed to warm models on multiple GPUs")
            
            logger.info("✅ Multi-GPU warming complete!")

    @fal.endpoint("/generate")
    def generate(self, input: GenerateInput) -> GenerateOutput:
        """
        Generate a video from a reference image using the pre-loaded GameCraft model.
        Much faster since model loading is done in setup().
        """
        print("🚀 Starting fast video generation...")
        
        # Use FAL's persistent directory for outputs that need to be served
        import uuid
        generation_id = str(uuid.uuid4())
        persistent_output_dir = Path(FAL_PERSISTENT_DIR) / "outputs" / generation_id
        persistent_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory only for intermediate files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Download input image using fal.toolkit
            print("📥 Downloading reference image...")
            img_path = tmp_path / "reference.png"
            downloaded_file = download_file(input.image_url, target_dir=str(tmp_path))
            
            # Copy or move the downloaded file to our expected location
            if Path(downloaded_file).name != "reference.png":
                shutil.copy2(downloaded_file, img_path)
            else:
                img_path = Path(downloaded_file)
            
            print(f"✓ Downloaded reference image to: {img_path}")
            
            # Create temporary output directory
            temp_output_dir = tmp_path / "results"
            temp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare input data with image path instead of URL
            input_data = {
                "image_path": str(img_path),
                "output_dir": str(temp_output_dir),
                "prompt": input.prompt,
                "negative_prompt": input.negative_prompt,
                "actions": input.actions,
                "size": input.size,
                "seed": input.seed,
                "guidance_scale": input.guidance_scale,
                "infer_steps": input.infer_steps,
                "use_fp8": input.use_fp8,
                "cpu_offload": input.cpu_offload,
            }
            
            # Use the generate_video function from the cloned repository
            result = self.generate_video(input_data, self.model_warmer)
            
            # Copy the video to persistent storage
            temp_video_path = Path(result["video_path"])
            persistent_video_path = persistent_output_dir / "generated_video.mp4"
            shutil.copy2(temp_video_path, persistent_video_path)
            
            print(f"✓ Video copied to persistent storage: {persistent_video_path}")
            
            return GenerateOutput(
                video=Video.from_path(str(persistent_video_path)),
                height=result["height"],
                width=result["width"],
                total_frames=result["total_frames"],
                log=result["log"]
            )
