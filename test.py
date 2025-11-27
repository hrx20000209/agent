from MobileAgentE.api import inference_chat_llama_cpp
from MobileAgentE.agents import OneStepAgent


def get_reasoning_response(chat):
    """唯一的 LLM 调用"""
    temperature = 0.0
    return inference_chat_llama_cpp(chat, temperature=temperature)

if __name__ == '__main__':
    agent = OneStepAgent(adb_path="adb")
    w, h = 1080, 1080
    history = []
    screenshot_path = "../../Desktop/screenshot.png"
    instruction = "See what is in the screen?"
    agent.run_step(
        instruction,
        screenshot_path,
        w, h,
        history=history,
        llm_api_func=get_reasoning_response
    )
