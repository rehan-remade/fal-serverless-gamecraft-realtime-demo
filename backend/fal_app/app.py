import os
import shutil
import subprocess
import tempfile
from pathlib import Path

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
    keep_alive=600,
):  # type: ignore
    machine_type = "GPU-H100"
    num_gpus = 1
    local_python_modules = ["fal_app"]
    requirements = ["huggingface_hub"]

    def setup(self):
        """
        Clone the repository and download model weights from Hugging Face.
        This runs once per warm container.
        """
        from huggingface_hub import hf_hub_download, snapshot_download

        # Clone the Hunyuan GameCraft repository
        self.repo_dir = Path(
            clone_repository(
                "https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0.git",
                target_dir="/tmp",
                include_to_path=True,
                repo_name="hunyuan-gamecraft",
            )
        )

        # Use /data for persistent model storage
        self.models_base_dir = Path("/data/models/hunyuan-gamecraft")
        self.models_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up model directories in /data
        self.gamecraft_models_dir = self.models_base_dir / "gamecraft_models"
        self.stdmodels_dir = self.models_base_dir / "stdmodels"
        
        # Create all model directories
        self.gamecraft_models_dir.mkdir(parents=True, exist_ok=True)
        
        # VAE directory
        self.vae_dir = self.stdmodels_dir / "vae_3d" / "hyvae"
        self.vae_dir.mkdir(parents=True, exist_ok=True)
        
        # CLIP directory
        self.clip_dir = self.stdmodels_dir / "openai_clip-vit-large-patch14"
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        
        # LLaVA directory
        self.llava_dir = self.stdmodels_dir / "llava-llama-3-8b-v1_1-transformers"
        self.llava_dir.mkdir(parents=True, exist_ok=True)

        # Create symbolic link in repo directory to point to /data models
        weights_link = self.repo_dir / "weights"
        if weights_link.exists() and weights_link.is_symlink():
            weights_link.unlink()
        elif weights_link.exists():
            shutil.rmtree(weights_link)
        weights_link.symlink_to(self.models_base_dir)

        # Enable HF transfer for faster downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["HF_HOME"] = "/data/.cache/huggingface"
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "1000000000000"
        os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "32"
        
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
                    print(f"✓ VAE file already exists: {target_vae}")

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

            print("\n=== All models ready! ===")
            
            # Set checkpoint path
            self.checkpoint_path = distilled_checkpoint_path
            
            # Verify checkpoint exists
            if not self.checkpoint_path.exists():
                raise RuntimeError(f"Checkpoint not found at {self.checkpoint_path}")
            
            print(f"✓ Using checkpoint: {self.checkpoint_path}")

        except Exception as e:
            print(f"Error downloading models: {e}")
            raise RuntimeError(f"Failed to download required model files: {e}")

    @fal.endpoint("/generate")
    def generate(self, input: GenerateInput) -> GenerateOutput:
        """
        Generate a video from a reference image using GameCraft.
        
        1) Download the reference image
        2) Run the GameCraft generation script
        3) Return the generated video
        """
        # Create temporary directory for this generation
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Download input image - using requests for simplicity
            img_path = tmp_path / "reference.png"
            downloaded_file = download_file(input.image_url, target_dir=str(tmp_path))
            
            # Copy or move the downloaded file to our expected location
            if Path(downloaded_file).name != "reference.png":
                shutil.copy2(downloaded_file, img_path)
            else:
                img_path = Path(downloaded_file)
            
            # Set up results directory
            results_dir = tmp_path / "results_distill"
            results_dir.mkdir(parents=True, exist_ok=True)

            # Extract parameters
            H, W = input.size
            action_ids = " ".join(a.id for a in input.actions)
            action_speeds = " ".join(str(a.speed) for a in input.actions)
            total_frames = sum(a.frames for a in input.actions)

            # Build command for single-GPU execution
            cmd = [
                "torchrun",
                "--nnodes=1",
                "--nproc_per_node=1",  # Single GPU for serverless
                "--master_port", "29605",
                "hymm_sp/sample_batch.py",
                "--image-path", str(img_path),
                "--prompt", input.prompt,
                "--add-neg-prompt", input.negative_prompt or "",
                "--ckpt", str(self.checkpoint_path),  # Use the checkpoint from /data
                "--video-size", str(H), str(W),
                "--cfg-scale", str(input.guidance_scale),
                "--image-start",
                "--action-list", *action_ids.split(),
                "--action-speed-list", *action_speeds.split(),
                "--seed", str(input.seed) if input.seed is not None else "250160",
                "--infer-steps", str(input.infer_steps),
                "--flow-shift-eval-video", "5.0",
                "--save-path", str(results_dir),
                "--sample-n-frames", str(total_frames),
            ]

            # Add optional flags
            if input.use_fp8:
                cmd.append("--use-fp8")
            if input.cpu_offload:
                cmd.append("--cpu-offload")

            # Set up environment
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{self.repo_dir}:{env.get('PYTHONPATH', '')}"
            env["MODEL_BASE"] = str(self.stdmodels_dir)  # Point to /data models
            env["HF_HOME"] = "/data/.cache/huggingface"
            
            # Run generation
            print(f"Running command: {' '.join(cmd)}")
            print(f"Checkpoint path: {self.checkpoint_path}")
            print(f"MODEL_BASE: {self.stdmodels_dir}")
            
            proc = subprocess.run(
                cmd,
                cwd=str(self.repo_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )
            log_text = proc.stdout or ""
            
            # Log the output for debugging
            if proc.returncode != 0:
                print(f"Command failed with return code {proc.returncode}")
                print(f"Output: {log_text}")

            # Find output video
            mp4_files = sorted(results_dir.glob("*.mp4"))
            if not mp4_files:
                # Check alternative output location
                alt_results = self.repo_dir / "results_distill"
                if alt_results.exists():
                    mp4_files = sorted(alt_results.glob("*.mp4"))
            
            if not mp4_files:
                raise RuntimeError(f"No video generated. Check logs:\n{log_text}")

            # Return the video using fal's Video class
            video_path = mp4_files[-1]
            
            return GenerateOutput(
                video=Video.from_path(video_path),
                height=H,
                width=W,
                total_frames=total_frames,
                log=log_text[-8000:] if log_text else "No logs available"
            )
