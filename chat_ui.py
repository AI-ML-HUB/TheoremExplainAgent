import os
import asyncio
from dotenv import load_dotenv
import gradio as gr

from generate_video import VideoGenerator
from mllm_tools.litellm import LiteLLMWrapper

load_dotenv()

# Initialize the LiteLLM models
planner_model = LiteLLMWrapper(
    model_name="gemini/gemini-1.5-pro-002",
    temperature=0.7,
    print_cost=True,
)
scene_model = planner_model
helper_model = planner_model

# Create a VideoGenerator instance
video_generator = VideoGenerator(
    planner_model=planner_model,
    scene_model=scene_model,
    helper_model=helper_model,
)

async def generate_video_for_theorem(theorem: str) -> str:
    """Generate a video for the given theorem and return the file path."""
    topic = theorem
    description = theorem
    await video_generator.generate_video_pipeline(topic, description)
    video_generator.combine_videos(topic)
    file_prefix = topic.lower()
    file_prefix = "".join(c if c.isalnum() else "_" for c in file_prefix)
    video_path = os.path.join(video_generator.output_dir, file_prefix, f"{file_prefix}_combined.mp4")
    return video_path

def chat(message: str, history: list[tuple[str, str]]):
    """Handle chat messages and trigger video generation."""
    video_path = asyncio.run(generate_video_for_theorem(message))
    history = history + [(message, f"Generated video saved to {video_path}")]
    return "", history

if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat,
        title="TheoremExplainAgent Chat",
        textbox=gr.Textbox(placeholder="Ask about a theorem here"),
    ).launch()
