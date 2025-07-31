import argparse
import json
import copy
from typing import Any, Optional

import tiktoken
from beartype import beartype
from PIL import Image

from agent.prompts import *

from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_stop_action,
    map_OS_Atlas_to_benchmark_action
)
from browser_env.utils import Observation, StateInfo
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer

from qwen_vl_utils import process_vision_info


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        local_llm: tuple | None,
        action_set_tag: str,
        lm_config: Optional[lm_config.LMConfig],
        prompt_constructor: PromptConstructor | LocalPromptConstructor,
        captioning_fn = None,
        self_training: bool = False
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn
        self.local_llm = local_llm
        self.self_training = self_training

        if self.local_llm is None:
            # Check if the model is multimodal.
            if ("gemini" in lm_config.model or "gpt-4" in lm_config.model and "vision" in lm_config.model) and type(prompt_constructor) == MultimodalCoTPromptConstructor:
                self.multimodal_inputs = True
            else:
                self.multimodal_inputs = False
        else:
            self.multimodal_inputs = True

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any], images: Optional[list[Image.Image]] = None,
        image_paths: Optional[list[str]] = None,
        output_response: bool = False
    ) -> Any:
        
        if self.local_llm is None:
            # Create page screenshot image for multimodal models.
            if self.multimodal_inputs:
                page_screenshot_arr = trajectory[-1]["observation"]["image"]
                page_screenshot_img = Image.fromarray(
                    page_screenshot_arr
                )  # size = (viewport_width, viewport_width)

            # Caption the input image, if provided.
            if images is not None and len(images) > 0:
                if self.captioning_fn is not None:
                    image_input_caption = ""
                    for image_i, image in enumerate(images):
                        if image_i == 0:
                            image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                        else:
                            image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                        if len(images) > 1:
                            image_input_caption += ", "
                    # Update intent to include captions of input images.
                    intent = f"{image_input_caption}\nIntent: {intent}"
                elif not self.multimodal_inputs:
                    print(
                        "WARNING: Input image provided but no image captioner available."
                    )

            # meta_data 里面存的是 action_history
            if self.multimodal_inputs:
                prompt = self.prompt_constructor.construct(
                    trajectory, intent, page_screenshot_img, images, meta_data
                )
            else:
                prompt = self.prompt_constructor.construct(
                    trajectory, intent, meta_data
                )
            lm_config = self.lm_config
            n = 0
            while True:
                response = call_llm(lm_config, prompt)
                force_prefix = self.prompt_constructor.instruction[
                    "meta_data"
                ].get("force_prefix", "")
                response = f"{force_prefix}{response}"
                if output_response:
                    print(f'Agent: {response}', flush=True)
                n += 1
                try:
                    parsed_response = self.prompt_constructor.extract_action(
                        response
                    )
                    if self.action_set_tag == "id_accessibility_tree":
                        action = create_id_based_action(parsed_response)
                    elif self.action_set_tag == "playwright":
                        action = create_playwright_action(parsed_response)
                    elif self.action_set_tag == "som":
                        action = create_id_based_action(parsed_response)
                    else:
                        raise ValueError(
                            f"Unknown action type {self.action_set_tag}"
                        )
                    action["raw_prediction"] = response
                    break
                except ActionParsingError as e:
                    if n >= lm_config.gen_config["max_retry"]:
                        action = create_none_action()
                        action["raw_prediction"] = response
                        break
        # 开始编写我们的模型产生 next_action 的代码
        else:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            page_screenshot_img = Image.fromarray(
                    page_screenshot_arr
                )
            
            # Caption the input image, if provided.
            if images and len(image_paths) > 0:
                with open("image_captions.json", "r", encoding="utf-8") as f:
                    image_captions = json.load(f)
                image_input_caption = ""
                for image_i, image_path in enumerate(image_paths):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{image_captions[image_path]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{image_captions[image_path]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
                    
            messages = self.prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images, meta_data
            )

            # with open("messages.json", "w", encoding="utf-8") as f:
            #     json.dump(messages, f, ensure_ascii=False, indent=4)
            # print("messages 已保存")

            response = get_response_from_localLLM(self.local_llm, messages, self.self_training)

            print("Agent:\n", response, flush=True)

            action_splitter = self.prompt_constructor.instruction["meta_data"]["action_splitter"]
            action_list = [extract_actions(action_splitter, response)] # 变成列表

            action_strs = copy.deepcopy(action_list)

            for i, action in enumerate(action_list):
                action_low_level = create_id_based_action(action)
                action_low_level["raw_prediction"] = response
                action_list[i] = action_low_level

        return action_strs, action_list

    def reset(self, test_config_file: str) -> None:
        pass

def get_response_from_localLLM(local_llm, messages, self_training):
    model, processor = local_llm
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    max_new_tokens = 512
    # Generate output
    if not self_training:
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    else:
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=1.0,
            top_p=0.9,
            top_k=50,
            do_sample=True
        )   # 改变输出策略，倾向于更加多样性的输出


    # Post-process the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    return output_text[0]

def extract_actions(action_splitter, response) -> str:
    # find the first occurence of action
    pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()
    else:
        raise ActionParsingError(
            f'Cannot find the action in "{response}"'
        )

def construct_agent(args: argparse.Namespace, local_llm=None, captioning_fn=None) -> Agent:
    llm_config = lm_config.construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        
        if args.provider != "local":
            tokenizer = Tokenizer(args.provider, args.model)
            prompt_constructor = eval(constructor_type)(
                args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
            )
            agent = PromptAgent(
                action_set_tag=args.action_set_tag,
                lm_config=llm_config,
                prompt_constructor=prompt_constructor,
                captioning_fn=captioning_fn,
            )
        else:
            prompt_constructor = eval(constructor_type)(
                args.instruction_path
            )
            agent = PromptAgent(
                local_llm=local_llm,
                action_set_tag=args.action_set_tag,
                lm_config=llm_config,
                prompt_constructor=prompt_constructor,
                captioning_fn=captioning_fn,
                self_training=args.self_training
            )

    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent

