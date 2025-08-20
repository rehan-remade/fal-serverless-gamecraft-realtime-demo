from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from fal.toolkit import Video


class Action(BaseModel):
    id: Literal["w", "a", "s", "d"]
    speed: float = 0.2
    frames: int = 33  # sample_n_frames per action


class GenerateInput(BaseModel):
    image_url: str = Field(
        description="URL of the reference image to use for generation"
    )
    prompt: str = Field(
        default="A charming medieval village ...",
        description="Text prompt describing the desired video"
    )
    negative_prompt: Optional[str] = Field(
        default="overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
        description="Negative prompt to guide what to avoid in generation"
    )
    actions: List[Action] = Field(
        default_factory=lambda: [
            Action(id="w", speed=0.2, frames=33),
            Action(id="s", speed=0.2, frames=33),
            Action(id="d", speed=0.2, frames=33),
            Action(id="a", speed=0.2, frames=33),
        ],
        description="List of camera movement actions for the video"
    )
    size: tuple[int, int] = Field(
        default=(704, 1216),
        description="Output video size (height, width) - must be multiples of 16"
    )
    infer_steps: int = Field(
        default=8,
        description="Number of inference steps (distilled fast path)"
    )
    guidance_scale: float = Field(
        default=1.0,
        description="Guidance scale for generation"
    )
    seed: Optional[int] = Field(
        default=250160,
        description="Random seed for reproducible generation"
    )
    use_fp8: bool = Field(
        default=False,  # Changed from True to False
        description="Use FP8 precision for faster inference"
    )
    cpu_offload: bool = Field(
        default=True,
        description="Offload models to CPU to save GPU memory"
    )


class GenerateOutput(BaseModel):
    video: Video = Field(
        description="Generated video file"
    )
    height: int = Field(
        description="Height of the generated video"
    )
    width: int = Field(
        description="Width of the generated video"
    )
    total_frames: int = Field(
        description="Total number of frames in the generated video"
    )
    log: str = Field(
        description="Generation log for debugging"
    )