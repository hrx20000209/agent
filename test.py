import os
from MobileAgentE.api import inference_chat_llama_cpp_xml_only
from MobileAgentE.agents import OneStepAgent_XML
from MobileAgentE.tree import parse_a11y_tree
from MobileAgentE.agents import InfoPool


########################################
# LLM Wrapper
########################################
def get_reasoning_response(chat):
    """唯一的 LLM 调用"""
    return inference_chat_llama_cpp_xml_only(chat, temperature=0.0)


########################################
# Test Code
########################################
if __name__ == "__main__":
    # ======== 1. 加载 XML 文件 ========
    xml_file = "./screenshot/a11y.xml"   # ❗换成你的 XML
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"{xml_file} not found.")

    with open(xml_file, "r", encoding="utf-8") as f:
        xml_str = f.read()

    print("Loaded XML length:", len(xml_str))

    # ======== 2. 解析 XML tree（给 Agent 用） ========
    tree = parse_a11y_tree(xml_file)

    # ======== 3. 创建 Agent ========
    agent = OneStepAgent_XML(adb_path="adb")  # adb 不会被用到

    # Fake 屏幕参数（你可以调整）
    w, h = 1080, 2400
    history = []
    instruction = "Based on this UI XML tree, determine the next action."

    # ======== 4. 调用 run_step（Text-only 模式） ========
    print("\n==== Calling OneStepAgent_XML.run_step ====\n")

    action = agent.run_step(
        instruction=instruction,
        screenshot_img=None,    # ❗XML-only，不再需要 screenshot
        width=w,
        height=h,
        history=history,
        llm_api_func=get_reasoning_response,
        xml_str=xml_str         # ❗关键：把 XML 传进去
    )

    # ======== 5. 打印 LLM Action ========
    print("\n==== LLM OUTPUT ACTION ====")
    print(action)
