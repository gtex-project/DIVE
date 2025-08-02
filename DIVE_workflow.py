from langgraph.graph import StateGraph, END, START
from IPython.display import Image, display
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
import os
from keys import set_api
import json
import base64
from json_tool import try_parse_json
from prompt_template import get_text_only_prompt, get_image_pct_prompt, get_image_discharge_prompt, get_image_tpd_prompt, get_text_image_prompt
from collections import Counter
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import glob
import time
import concurrent.futures
import argparse

# =========================
#   USER CONFIGURABLES
# =========================
USER_CONFIG = {
    "DOI_CSV": "/workspace/GT_results/ground_truth.csv",
    "PDF_MINERU_SOURCES": [
        "./MinerU/MinerU_Outputs-1",
    ],
    "OUTPUT_CSV": "/workspace/PaperReading_Output/20250721_100gt_papers/Qwen2.5-VL-72B-Instruct_deepseek-qwen3-8b-2step.csv",
    "SYSTEM_MESSAGE": "Extract information from the literature and save it in JSON format. Think and provide your data extracted.",
    "TWO_STEP_EXTRACTION": True,
    "MAX_WORKERS": 16,
    "SAVE_EVERY": 16,
    "MAX_IMAGE_NUM": 5,
    "MAX_TOKEN_LIMIT": 40000,
    "LLM_GRAPH_CONFIG": {
        "openai_api_base": "https://api.siliconflow.cn/v1",
        "openai_api_key": os.environ.get("SILICONFLOW_API_KEY", ""),
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "max_tokens": 10000
    },
    "LLM_GRAPH_TYPE_CONFIG": {
        "openai_api_base": "https://h200.digcat.org/v1",
        "openai_api_key": "sk-haolilabvllm",
        "model": "deepseek-r1-qwen3-8b",
        "temperature": 0.6,
        "top_p": 0.95,
        "timeout": 3600,
        "max_tokens": 30000
    },
    "LLM_TEXT_CONFIG": {
        "openai_api_base": "https://h200.digcat.org/v1",
        "openai_api_key": "sk-haolilabvllm",
        "model": "deepseek-r1-qwen3-8b",
        "temperature": 0.6,
        "top_p": 0.95,
        "timeout": 3600,
        "max_tokens": 30000
    }
}

# =========================
#   LLM INITIALIZATION
# =========================
set_api("google")
set_api("siliconflow")
set_api("deepseek")
set_api("groq")
set_api("openai")
set_api("claude")

llm_graph = ChatOpenAI(**USER_CONFIG["LLM_GRAPH_CONFIG"])
llm_graph_type = ChatOpenAI(**USER_CONFIG["LLM_GRAPH_TYPE_CONFIG"])
llm_text = ChatOpenAI(**USER_CONFIG["LLM_TEXT_CONFIG"])

# =========================
#   KEYWORD LISTS
# =========================
PCT_KEYWORDS = [
    "PCT", "P–C isotherm", "PCI", "P–C–T", "P-C-T", "PeC", "p-C-T", "p–c–T",
    "C-T", "Pressure-composition", "Pec isotherms", "P–C", "P-C", "PeC-T",
]
ELEC_KEYWORDS = [
    "discharge capacity", "discharge"
]
TPD_KEYWORDS = [
    "TPD", "isotherm", "dehydrogenation", "hydrogenation", "Absorption", "Desorption", "adsorption"
]

# =========================
#   WORKFLOW STATE
# =========================
class State(MessagesState):
    doi: str  
    paper_input_path: str
    paper_json: list[str]
    prompt: str
    figure_type: str 
    figure_caption_idx: dict
    thinking: str
    image_flag: bool

# =========================
#   UTILITY FUNCTIONS
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Process scientific papers for data extraction.")
    parser.add_argument('--doi_csv', type=str, default=USER_CONFIG["DOI_CSV"], help='Path to DOI CSV file')
    parser.add_argument('--pdf_sources', type=str, nargs='+', default=USER_CONFIG["PDF_MINERU_SOURCES"], help='List of PDF MinerU source directories')
    parser.add_argument('--output_csv', type=str, default=USER_CONFIG["OUTPUT_CSV"], help='Path to output CSV file')
    parser.add_argument('--system_message', type=str, default=USER_CONFIG["SYSTEM_MESSAGE"], help='System message for extraction')
    parser.add_argument('--two_step', action='store_true', default=USER_CONFIG["TWO_STEP_EXTRACTION"], help='Use two-step extraction')
    parser.add_argument('--max_worker', type=int, default=USER_CONFIG["MAX_WORKERS"], help='Number of parallel workers')
    parser.add_argument('--save_every', type=int, default=USER_CONFIG["SAVE_EVERY"], help='Save every N records')
    args, unknown = parser.parse_known_args()
    return args

def workflow_selection(state):
    if state["image_flag"]:
        if state["figure_type"] == "PCT":
            return "read_graph_pct"
        if state["figure_type"] == "TPD":
            return "read_tpd_or_isotherm"
        if state["figure_type"] == "ELEC":
            return "read_graph_elec"
        else:
            return "text_data_extraction_onestep"
    else:
        return "text_data_extraction_onestep"

def generate_caption_context(image_list):
    context = ""
    for i, image in enumerate(image_list):
        caption = image['img_caption'][0]
        context_text = '\n'.join(image['before_context'] + image['after_context'])
        context += f"Image {i + 1}'s caption:\n{caption}\n\nImage {i + 1}'s context:\n{context_text}\n\n"
    return context   

def base64_image(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_uri = f"data:image/jpeg;base64,{image_base64}"
    return image_data_uri

def generate_image_prompt(state, figure_caption_dict=None):
    IMAGE_CONTEXT_LENGTH = 5
    image_paths = []
    if state["figure_type"] == "PCT":
        image_prompt = get_image_pct_prompt()
    elif state["figure_type"] == "ELEC":
        image_prompt = get_image_discharge_prompt()
    elif state["figure_type"] == "TPD":
        image_prompt = get_image_tpd_prompt()
    with open(state["paper_input_path"], "r") as fr:
        json_str = fr.read()
    content_list = json.loads(json_str)
    target_image = []
    for idx in figure_caption_dict[state["figure_type"]]:
        content_dict = content_list[idx]
        if content_dict["type"] == "image":
            if len(content_dict['img_caption']) > 0:
                if content_dict['img_path'] != "":
                    image_paths.append(content_dict['img_path'])
                context_num = 1
                content_i = 1
                content_dict["before_context"] = []
                content_dict["after_context"] = []
                while True:
                    if context_num > IMAGE_CONTEXT_LENGTH or idx - content_i < 0:
                        break
                    if content_list[idx - content_i]["type"] == "text":
                        content_dict["before_context"].insert(0, content_list[idx - content_i]["text"])
                        context_num += 1
                    content_i += 1    
                context_num = 1
                content_i = 1
                while True:
                    if context_num > IMAGE_CONTEXT_LENGTH or idx + content_i + 1 > len(content_list):
                        break
                    if content_list[idx + content_i]["type"] == "text":
                        content_dict["after_context"].append(content_list[idx + content_i]["text"])
                        context_num += 1   
                    content_i += 1    
                target_image.append(content_dict)
    formatted_prompt = image_prompt.format(caption_context=generate_caption_context(target_image))
    return formatted_prompt, content_list, image_paths

def read_graph_pct(state):
    state["figure_type"] = "PCT"
    auto_dir = os.path.dirname(state["paper_input_path"])
    prompt, paper_content_list, image_paths = generate_image_prompt(state, figure_caption_dict=state["figure_caption_idx"])
    input_message = [{"type": "text", "text": prompt}]
    if len(image_paths) > 0:
        if len(image_paths) > USER_CONFIG["MAX_IMAGE_NUM"]:
            image_paths = image_paths[:USER_CONFIG["MAX_IMAGE_NUM"]]
        for image in image_paths:
            if os.path.exists(os.path.join(auto_dir, image)):
                input_message.append({"type": "image_url", "image_url": {"url": base64_image(os.path.join(auto_dir, image))}})
            else:
                with open("missing_graph.txt", "a") as fw:
                    fw.writelines([f"{os.path.join(auto_dir, image)}\n"])
    input_message = state["messages"] + [HumanMessage(content=input_message)]
    raw_response = llm_graph.invoke(input_message)
    output_state = {
        "doi": state["doi"],
        "messages": [raw_response],
        "figure_type": state["figure_type"],
        "paper_input_path": state["paper_input_path"],
        "paper_json": paper_content_list,
    }
    return output_state

def read_graph_elec(state):
    state["figure_type"] = "ELEC"
    auto_dir = os.path.dirname(state["paper_input_path"])
    prompt, paper_content_list, image_paths = generate_image_prompt(state, figure_caption_dict=state["figure_caption_idx"])
    input_message = [{"type": "text", "text": prompt}]
    if len(image_paths) > 0:
        if len(image_paths) > USER_CONFIG["MAX_IMAGE_NUM"]:
            image_paths = image_paths[:USER_CONFIG["MAX_IMAGE_NUM"]]
        for image in image_paths:
            if os.path.exists(os.path.join(auto_dir, image)):
                input_message.append({"type": "image_url", "image_url": {"url": base64_image(os.path.join(auto_dir, image))}})
            else:
                with open("missing_graph.txt", "a") as fw:
                    fw.writelines([f"{os.path.join(auto_dir, image)}\n"])
    input_message = state["messages"] + [HumanMessage(content=input_message)]
    raw_response = llm_graph.invoke(input_message)
    output_state = {
        "doi": state["doi"],
        "messages": [raw_response],
        "figure_type": state["figure_type"],
        "paper_input_path": state["paper_input_path"],
        "paper_json": paper_content_list,
    }
    return output_state

def read_tpd_or_isotherm(state):
    state["figure_type"] = "TPD"
    auto_dir = os.path.dirname(state["paper_input_path"])
    prompt, paper_content_list, image_paths = generate_image_prompt(state, figure_caption_dict=state["figure_caption_idx"])
    input_message = [{"type": "text", "text": prompt}]
    if len(image_paths) > 0:
        if len(image_paths) > USER_CONFIG["MAX_IMAGE_NUM"]:
            image_paths = image_paths[:USER_CONFIG["MAX_IMAGE_NUM"]]
        for image in image_paths:
            if os.path.exists(os.path.join(auto_dir, image)):
                input_message.append({"type": "image_url", "image_url": {"url": base64_image(os.path.join(auto_dir, image))}})
            else:
                with open("missing_graph.txt", "a") as fw:
                    fw.writelines([f"{os.path.join(auto_dir, image)}\n"])
    input_message = state["messages"] + [HumanMessage(content=input_message)]
    raw_response = llm_graph.invoke(input_message)
    output_state = {
        "doi": state["doi"],
        "messages": [raw_response],
        "figure_type": state["figure_type"],
        "paper_input_path": state["paper_input_path"],
        "paper_json": paper_content_list,
    }
    return output_state

def trim_paper(state, included_base64_image=False):
    import re
    paper_json_path = state["paper_input_path"]
    md_path = paper_json_path.rstrip("_content_list.json") + ".md"
    if not os.path.exists(md_path):
        return {"text": "", "images": {}}
    filter_titles = {"introduction", "method", "references", "acknowledgment", "author"}
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sections = []
    current_title = None
    current_content = []
    for line in lines:
        if line.strip().startswith("#"):
            if current_title is not None:
                sections.append((current_title, current_content))
            title_text = line.strip().lstrip("#").strip().lower()
            current_title = title_text
            current_content = [line]
        else:
            current_content.append(line)
    if current_title is not None:
        sections.append((current_title, current_content))
    filtered_sections = [
        content for title, content in sections if title not in filter_titles
    ]
    final_lines = []
    md_dir = os.path.dirname(md_path)
    image_pattern = re.compile(r'!\[\]\((images/[^)]+)\)')
    image_dict = {}
    for section in filtered_sections:
        for line in section:
            match = image_pattern.search(line)
            if included_base64_image and match:
                img_rel_path = match.group(1)
                img_abs_path = os.path.join(md_dir, img_rel_path)
                if os.path.exists(img_abs_path):
                    with open(img_abs_path, "rb") as img_f:
                        img_bytes = img_f.read()
                        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                        ext = os.path.splitext(img_abs_path)[1][1:] or "jpg"
                        if ext == "jpg":
                            mime = "jpeg"
                        elif ext == "png":
                            mime = "png"
                        elif ext == "jpeg":
                            mime = "jpeg"
                        elif ext == "webp":
                            mime = "webp"
                        else:
                            mime = ext
                        url = f"data:image/{mime};base64,{img_b64}"
                        image_dict[img_rel_path] = url
                else:
                    image_dict[img_rel_path] = None
            final_lines.append(line)
    final_text = "".join(final_lines)
    return {"text": final_text, "images": image_dict}

def text_data_extraction(state):
    text_prompt = get_text_image_prompt()
    input_prompt = text_prompt.format(paper_text=trim_paper(state)["text"], image_data=state["messages"][-1].content)
    input_message = [HumanMessage(content=input_prompt)]
    if llm_text.get_num_tokens(input_prompt) > USER_CONFIG["MAX_TOKEN_LIMIT"]:
        output_state = {
            "doi": state["doi"],
            "messages": [SystemMessage(content="Maxium token limited.")],
            "summary": "Error: Maxium token limited.",
            "paper_input_path": state["paper_input_path"],
            "paper_json": state["paper_json"],
            "prompt": input_prompt,
            "figure_type": state["figure_type"],
            "thinking": "Error: Maxium token limited.",
            "image_flag": state["image_flag"]
        }
        return output_state
    best_response = None
    best_list_len = -1
    responses = []
    for _ in range(2):
        raw_response = llm_text.invoke(input_message)
        try:
            parsed = try_parse_json(raw_response.content)
        except Exception:
            parsed = False
        if isinstance(parsed, list) and parsed:
            responses.append((len(parsed), raw_response))
            if len(parsed) > best_list_len:
                best_list_len = len(parsed)
                best_response = raw_response
        else:
            responses.append((0, raw_response))
    if best_response is None:
        best_response = raw_response
    raw_response = best_response
    if "</think>" in raw_response.content:
        output_state = {
            "doi": state["doi"],
            "messages": [raw_response],
            "paper_input_path": state["paper_input_path"],
            "paper_json": state["paper_json"],
            "prompt": input_prompt,
            "figure_type": state["figure_type"],
            "thinking": raw_response.content.split("</think>")[0],
            "image_flag": state["image_flag"]
        }
    else:
        output_state = {
            "doi": state["doi"],
            "messages": [raw_response],
            "paper_input_path": state["paper_input_path"],
            "paper_json": state["paper_json"],
            "prompt": input_prompt,
            "figure_type": state["figure_type"],
            "thinking": "",
            "image_flag": state["image_flag"]
        }
    return output_state

def text_data_extraction_onestep(state):
    paper_dict = trim_paper(state, included_base64_image=True)
    text = paper_dict["text"]
    image_dict = paper_dict["images"]
    text_prompt = get_text_only_prompt(with_physical=False)
    input_prompt = text_prompt.format(paper_text=text)
    input_message = [{"type": "text", "text": input_prompt}]
    if len(image_dict) > 0:
        if len(image_dict) > USER_CONFIG["MAX_IMAGE_NUM"]:
            image_dict = dict(list(image_dict.items())[:USER_CONFIG["MAX_IMAGE_NUM"]])
        for image_path, image_url in image_dict.items():
            if image_url is not None:
                input_message.append({"type": "image_url", "image_url": {"url": image_url}})
            else:
                with open("missing_graph.txt", "a") as fw:
                    fw.writelines([f"{os.path.join(os.path.dirname(state['paper_input_path']), image_path)}\n"])
    input_message = state["messages"] + [HumanMessage(content=input_message)]
    if llm_graph.get_num_tokens(input_prompt) > USER_CONFIG["MAX_TOKEN_LIMIT"]:
        output_state = {
            "doi": state["doi"],
            "messages": [SystemMessage(content="Maxium token limited.")],
            "summary": "Error: Maxium token limited.",
            "paper_input_path": state["paper_input_path"],
            "paper_json": state["paper_json"],
            "prompt": input_prompt,
            "figure_type": state["figure_type"],
            "thinking": "Error: Maxium token limited.",
            "image_flag": state["image_flag"]
        }
        return output_state
    best_response = None
    best_list_len = -1
    responses = []
    while len(responses) < 1:
        raw_response = llm_graph.invoke(input_message)
        try:
            parsed = try_parse_json(raw_response.content)
        except Exception:
            parsed = False
        if isinstance(parsed, list) and parsed:
            responses.append((len(parsed), raw_response))
            if len(parsed) > best_list_len:
                best_list_len = len(parsed)
                best_response = raw_response
    if best_response is None:
        best_response = raw_response
    raw_response = best_response
    if "</think>" in raw_response.content:
        output_state = {
            "doi": state["doi"],
            "messages": [raw_response],
            "summary": raw_response.content.split("</think>")[1].strip(),
            "paper_input_path": state["paper_input_path"],
            "paper_json": state["paper_json"],
            "prompt": input_prompt,
            "figure_type": state["figure_type"],
            "thinking": raw_response.content.split("</think>")[0],
            "image_flag": state["image_flag"]
        }
    else:
        output_state = {
            "doi": state["doi"],
            "messages": [raw_response],
            "summary": raw_response.content,
            "paper_input_path": state["paper_input_path"],
            "paper_json": state["paper_json"],
            "prompt": input_prompt,
            "figure_type": state["figure_type"],
            "thinking": "",
            "image_flag": state["image_flag"]
        }
    return output_state

config = {"configurable": {"thread_id": "1"}}
builder = StateGraph(State)
builder.add_conditional_edges(START, workflow_selection)
builder.add_node("read_graph_pct", read_graph_pct)
builder.add_node("text_data_extraction", text_data_extraction)
builder.add_node("text_data_extraction_onestep", text_data_extraction_onestep)
builder.add_node("read_graph_elec", read_graph_elec)
builder.add_node("read_tpd_or_isotherm", read_tpd_or_isotherm)
builder.add_edge("read_graph_pct", "text_data_extraction")
builder.add_edge("text_data_extraction", END)
builder.add_edge("text_data_extraction_onestep", END)
builder.add_edge("read_graph_elec", "text_data_extraction")
builder.add_edge("read_tpd_or_isotherm", "text_data_extraction")
graph = builder.compile()

def load_doi_and_outputs(doi_csv, output_csv):
    df_doi = pd.read_csv(doi_csv)
    doi_array = df_doi["doi"].values
    if os.path.exists(output_csv):
        print(f"Output file {output_csv} already exists. Loading existing data...")
        df_exists = pd.read_csv(output_csv)
        done_dois = set(df_exists["doi"].values)
    else:
        df_exists = None
        done_dois = set()
    return doi_array, df_exists, done_dois

def build_paper_json_path_list(doi_array, done_dois, pdf_mineru_sources):
    paper_json_path_list = []
    for doi in doi_array:
        if doi in done_dois:
            continue
        for pdf_path in pdf_mineru_sources:
            target_path = os.path.join(pdf_path, doi.replace("/", "_"))
            if os.path.exists(target_path):
                auto_path = os.path.join(target_path, "auto")
                json_files = glob.glob(os.path.join(auto_path, "*_content_list.json"))
                if json_files and os.path.isfile(json_files[0]):
                    paper_json_path_list.append(json_files[0])
                else:
                    with open("missing_pdf.txt", "a") as fw:
                        fw.writelines([doi + "\n"])
                    print(f"{doi} is missing a pdf json file.")
                break
    return paper_json_path_list

def process_paper(paper_json_path):
    doi = paper_json_path.split("/")[-3]
    figure_caption_dict = {"PCT": [], "ELEC": [], "TPD": []}
    try:
        with open(paper_json_path, "r", encoding='utf-8') as fr:
            json_str = fr.read()
        content_list = json.loads(json_str)
    except Exception as e:
        print(f"[{doi}] Failed to read JSON file: {str(e)}")
        return None
    figure_type = None
    for idx, content_dict in enumerate(content_list):
        if content_dict["type"] == "image" and len(content_dict['img_caption']) > 0:
            try:
                prompt = (
                    "You are given an image caption from a scientific paper. "
                    "Please determine which of the following categories the caption most likely belongs to, "
                    "based on the provided keywords for each category:\n\n"
                    "1. PCT-type (Pressure-Composition-Temperature isotherms):\n"
                    f"Caption Keywords may include: {', '.join(PCT_KEYWORDS)}\n\n"
                    "2. Electrochemical Discharge-type:\n"
                    f"Caption Keywords may include: {', '.join(ELEC_KEYWORDS)}\n\n"
                    "3. TPD or Isotherm-type (Temperature Programmed Desorption or Isotherm):\n"
                    f"Caption Keywords may include: {', '.join(TPD_KEYWORDS)}\n\n"
                    "Please answer with only the category number (1, 2, or 3). If the caption does not match any category, answer 0.\n\n"
                    "If multiple categories apply, the priority is 1 > 2 > 3\n\n"
                    f"Caption: {content_dict['img_caption'][0]}"
                )
                votes = []
                for _ in range(3):
                    response = llm_graph_type.invoke(prompt)
                    answer = response.content.split("</think>")[-1].strip()
                    votes.append(answer)
                majority_answer = Counter(votes).most_common(1)[0][0]
                answer = majority_answer
                if "1" in answer:
                    figure_caption_dict["PCT"].append(idx)
                if "2" in answer:
                    figure_caption_dict["ELEC"].append(idx)
                if "3" in answer:
                    figure_caption_dict["TPD"].append(idx)
                else:
                    figure_type = "TEXT"
            except Exception as e:
                print(f"[{doi}] image classification failed (idx {idx}): {str(e)}")
                continue
    if len(figure_caption_dict["PCT"]) > 0:
        figure_type = "PCT"
    elif len(figure_caption_dict["ELEC"]) > 0:
        figure_type = "ELEC"
    elif len(figure_caption_dict["TPD"]) > 0:
        figure_type = "TPD"
    elif len(figure_caption_dict["PCT"]) == 0 and len(figure_caption_dict["TPD"]) == 0 and len(figure_caption_dict["ELEC"]) == 0:
        figure_type = "TEXT"
    input_state = {
        "doi": doi,
        "messages": [SystemMessage(content=USER_CONFIG["SYSTEM_MESSAGE"])],
        "paper_input_path": paper_json_path,
        "paper_json": "",
        "prompt": "",
        "thinking": "",
        "figure_type": figure_type,
        "figure_caption_idx": figure_caption_dict,
        "image_flag": USER_CONFIG["TWO_STEP_EXTRACTION"]
    }
    max_retries = 1
    for attempt in range(max_retries):
        try:
            output = graph.invoke(input_state)
            messages = output.get("messages", [])
            if len(messages) > 2:
                image_data = messages[1].content if hasattr(messages[1], 'content') else "No image data"
                final_data = messages[2].content if hasattr(messages[2], 'content') else "No final data"
            elif len(messages) > 1:
                image_data = "No image data"
                final_data = messages[1].content if hasattr(messages[1], 'content') else "No final data"
            else:
                image_data = "No image data"
                final_data = "No final data"
            return {
                "doi": output["doi"].replace("_", "/"),
                "messages": output["messages"],
                "image_data": image_data,
                "final_data": final_data,
                "figure_type": output["figure_type"],
                "paper_input_path": output["paper_input_path"],
                "paper_json": output["paper_json"],
                "prompt": output["prompt"],
                "thinking": output["thinking"],
                "image_flag": output["image_flag"]
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt) 
    return None

def main(args=None):
    if args is None:
        args = parse_args()
    doi_array, df_exists, done_dois = load_doi_and_outputs(args.doi_csv, args.output_csv)
    paper_json_path_list = build_paper_json_path_list(doi_array, done_dois, args.pdf_sources)
    all_outputs = []
    counter = 0
    failed_count = 0
    total_tasks = len(paper_json_path_list)
    print(f"Starting processing of {total_tasks} papers with {args.max_worker} parallel threads.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_worker) as executor:
        future_to_path = {executor.submit(process_paper, path): path for path in paper_json_path_list}
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            doi = path.split("/")[-3]
            try:
                result = future.result(timeout=600)
                if result is not None:
                    all_outputs.append(result)
                    counter += 1
                    print(f"[SUCCESS] [{counter}/{total_tasks}] Processed: {doi}")
                else:
                    failed_count += 1
                    print(f"[FAIL] [{failed_count}] Failed: {doi}")
            except concurrent.futures.TimeoutError:
                failed_count += 1
                print(f"[TIMEOUT] [{failed_count}] Timeout: {doi}")
            except Exception as e:
                failed_count += 1
                print(f"[EXCEPTION] [{failed_count}] Exception: {doi} - {str(e)}")
            processed = counter + failed_count
            print(f"[Progress] {processed}/{total_tasks} processed (Success: {counter}, Failed: {failed_count})")
            if counter > 0 and counter % args.save_every == 0:
                try:
                    df_new = pd.DataFrame(all_outputs)
                    if df_exists is None:
                        df_new.to_csv(args.output_csv, index=False)
                    else:
                        df_combined = pd.concat([df_exists, df_new], ignore_index=True)
                        df_combined.to_csv(args.output_csv, index=False)
                    print(f"[Checkpoint] Saved {counter} records.")
                except Exception as e:
                    print(f"[WARNING] Failed to save: {str(e)}")
    print(f"[DONE] Processing complete! Total: {counter} success, {failed_count} failed.")
    df_new = pd.DataFrame(all_outputs)
    if df_exists is None:
        df_new.to_csv(args.output_csv, index=False)
    else:
        df_combined = pd.concat([df_exists, df_new], ignore_index=True)
        df_combined.to_csv(args.output_csv, index=False)
    print("[COMPLETE] All done and saved.")

if __name__ == "__main__":
    main()