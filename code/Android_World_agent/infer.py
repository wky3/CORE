# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import time
from typing import Any, Optional
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import numpy as np
from PIL import Image
import requests

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info

ERROR_CALLING_LLM = 'Error calling LLM'


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
  """Converts a numpy array into a byte string for a JPEG image."""
  image = Image.fromarray(image)
  return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
  in_mem_file = io.BytesIO()
  image.save(in_mem_file, format='JPEG')
  # Reset file pointer to start
  in_mem_file.seek(0)
  img_bytes = in_mem_file.read()
  return img_bytes


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Gemini GCP interface."""

  def __init__(
      self,
      model_name: str | None = None,
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.95,
      enable_safety_checks: bool = True,
  ):
    if 'GCP_API_KEY' not in os.environ:
      raise RuntimeError('GCP API key not set.')
    genai.configure(api_key=os.environ['GCP_API_KEY'])
    self.llm = genai.GenerativeModel(
        model_name,
        safety_settings=None
        if enable_safety_checks
        else SAFETY_SETTINGS_BLOCK_NONE,
        generation_config=generation_types.GenerationConfig(
            temperature=temperature, top_p=top_p, max_output_tokens=1000
        ),
    )
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)

  def predict(
      self,
      text_prompt: str,
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(
        text_prompt, [], enable_safety_checks, generation_config
    )

  def is_safe(self, raw_response):
    try:
      return (
          raw_response.candidates[0].finish_reason
          != answer_types.FinishReason.SAFETY
      )
    except Exception:  # pylint: disable=broad-exception-caught
      #  Assume safe if the response is None or doesn't have candidates.
      return True

  def predict_mm(
      self,
      text_prompt: str,
      images: list[np.ndarray],
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    counter = self.max_retry
    retry_delay = 1.0
    output = None
    while counter > 0:
      try:
        output = self.llm.generate_content(
            [text_prompt] + [Image.fromarray(image) for image in images],
            safety_settings=None
            if enable_safety_checks
            else SAFETY_SETTINGS_BLOCK_NONE,
            generation_config=generation_config,
        )
        return output.text, True, output
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2

    if (output is not None) and (not self.is_safe(output)):
      return ERROR_CALLING_LLM, False, output
    return ERROR_CALLING_LLM, None, None

  def generate(
      self,
      contents: (
          content_types.ContentsType | list[str | np.ndarray | Image.Image]
      ),
      safety_settings: safety_types.SafetySettingOptions | None = None,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Any]:
    """Exposes the generate_content API.

    Args:
      contents: The input to the LLM.
      safety_settings: Safety settings.
      generation_config: Generation config.

    Returns:
      The output text and the raw response.
    Raises:
      RuntimeError:
    """
    counter = self.max_retry
    retry_delay = 1.0
    response = None
    if isinstance(contents, list):
      contents = self.convert_content(contents)
    while counter > 0:
      try:
        response = self.llm.generate_content(
            contents=contents,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        return response.text, response
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2
    raise RuntimeError(f'Error calling LLM. {response}.')

  def convert_content(
      self,
      contents: list[str | np.ndarray | Image.Image],
  ) -> content_types.ContentsType:
    """Converts a list of contents to a ContentsType."""
    converted = []
    for item in contents:
      if isinstance(item, str):
        converted.append(item)
      elif isinstance(item, np.ndarray):
        converted.append(Image.fromarray(item))
      elif isinstance(item, Image.Image):
        converted.append(item)
    return converted


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI GPT4 wrapper.

  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      max_retry: int = 3,
      temperature: float = 0.0,
  ):
    if 'OPENAI_API_KEY' not in os.environ:
      raise RuntimeError('OpenAI API key not set.')
    self.openai_api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.model = model_name

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.openai_api_key}',
    }

    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
        'max_tokens': 1000,
    }

    # Gpt-4v supports multiple images, just need to insert them in the content
    # list.
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        response = requests.post(
            'https://api.openai-proxy.org/v1/chat/completions',  # 改成 CloseAI 的网址
            # 'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload,
        )
        print(response)
        if response.ok and 'choices' in response.json():
          return (
              response.json()['choices'][0]['message']['content'],
              None,
              response,
          )
        print(
            'Error calling OpenAI API with error message: '
            + response.json()['error']['message']
        )
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None

class Qwen2_5vlWrapper(LlmWrapper, MultimodalLlmWrapper):
    """Wrapper for GUI-R1 multimodal generation."""

    RETRY_WAIT_SECONDS = 5

    def __init__(
        self,
        base_model_path: str,
        lora_weights_path: str = None,
        self_training: bool = False,
        max_retry: int = 3,
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.torch_dtype=torch.float16
        self.max_pixels = 262144 # Inference 与 Train 保持一致
        # 重试次数限制
        self.max_retry = max(1, min(max_retry, 5))
        self.max_new_tokens = 128
        self.self_training = self_training

        # 加载模型
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path, 
            torch_dtype=self.torch_dtype, 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            base_model_path,
            max_pixels=self.max_pixels
        )

        if lora_weights_path is not None:
            # 加载 LoRA 权重并应用到基础模型
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights_path
            )

    def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
        # 纯文本调用
        return self.predict_mm(text_prompt, [])

    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        counter = self.max_retry
        retry_delay = self.RETRY_WAIT_SECONDS

        # Prepare the multimodal content for the model
        pil_images = [Image.fromarray(image) for image in images]

        # The content list alternates between text and image placeholders
        content = [{"type": "text", "text": text_prompt}]
        for img in pil_images:
            content.append({"type": "image", "image": img})
            
        messages = [{"role": "user", "content": content}]

        # import pickle
        # with open("messages.pkl", "wb") as f:
        #   pickle.dump(messages, f)

        # Format the prompt using the processor's chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process the text and images to get model inputs
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        while counter > 0:
            try:
                # Generate output
                if not self.self_training:
                    generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                else:
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=self.max_new_tokens,
                        temperature=1.0,
                        top_p=0.9,
                        top_k=50,
                        do_sample=True
                    )   # 改变输出策略，倾向于更加多样性的输出

                # Decode the generated tokens, skipping special tokens and the prompt
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                output_text = output_text[0]
                # return output_text[0], None, generated_ids_trimmed
                return output_text, None, output_text

            except Exception as e:
                counter -= 1
                print(f"Error calling LLM, will retry in {retry_delay} seconds. Retries left: {counter}", flush=True)
                print(e, flush=True)
                if counter > 0:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        return ERROR_CALLING_LLM, None, None


class Qwen2vlWrapper(LlmWrapper, MultimodalLlmWrapper):
    """Wrapper for OS-Atlas multimodal generation."""

    RETRY_WAIT_SECONDS = 5

    def __init__(
        self,
        base_model_path: str,
        lora_weights_path: str = None,
        self_training: bool = False,
        max_retry: int = 3,
    ):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.torch_dtype=torch.float16
        self.max_pixels = 262144 # Inference 与 Train 保持一致
        # 重试次数限制
        self.max_retry = max(1, min(max_retry, 5))
        self.max_new_tokens = 128
        self.self_training = self_training

        # 加载模型
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_path, 
            torch_dtype=self.torch_dtype, 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            base_model_path,
            max_pixels=self.max_pixels
        )

        if lora_weights_path is not None:
            # 加载 LoRA 权重并应用到基础模型
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights_path
            )

    def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
        # 纯文本调用
        return self.predict_mm(text_prompt, [])

    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        counter = self.max_retry
        retry_delay = self.RETRY_WAIT_SECONDS

        # Prepare the multimodal content for the model
        pil_images = [Image.fromarray(image) for image in images]

        # The content list alternates between text and image placeholders
        content = [{"type": "text", "text": text_prompt}]
        for img in pil_images:
            content.append({"type": "image", "image": img})
            
        messages = [{"role": "user", "content": content}]

        # import pickle
        # with open("messages.pkl", "wb") as f:
        #   pickle.dump(messages, f)

        # Format the prompt using the processor's chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process the text and images to get model inputs
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        while counter > 0:
            try:
                # Generate output
                if not self.self_training:
                    generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                else:
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=self.max_new_tokens,
                        temperature=1.0,
                        top_p=0.9,
                        top_k=50,
                        do_sample=True
                    )   # 改变输出策略，倾向于更加多样性的输出

                # Decode the generated tokens, skipping special tokens and the prompt
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                output_text = output_text[0]
                # return output_text[0], None, generated_ids_trimmed
                return output_text, None, output_text

            except Exception as e:
                counter -= 1
                print(f"Error calling LLM, will retry in {retry_delay} seconds. Retries left: {counter}", flush=True)
                print(e, flush=True)
                if counter > 0:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        return ERROR_CALLING_LLM, None, None
