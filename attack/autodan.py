import gc
import re
import sys
import time
import openai
import random


import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet

import torch
import numpy as np

from tqdm import tqdm
from utils import rank_indices, forward

from collections import defaultdict, OrderedDict

def autodan_sample_control(control_suffixs, score_list, num_elites, batch_size, crossover=0.5,
                           num_points=5, mutation=0.01, API_key=None, reference=None, if_softmax=True, if_api=True):
    score_list = [-x for x in score_list]
    # Step 1: Sort the score_list and get corresponding control_suffixs
    sorted_indices = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
    sorted_control_suffixs = [control_suffixs[i] for i in sorted_indices]

    # Step 2: Select the elites
    elites = sorted_control_suffixs[:num_elites]

    # Step 3: Use roulette wheel selection for the remaining positions
    parents_list = roulette_wheel_selection(control_suffixs, score_list, batch_size - num_elites, if_softmax)

    # Step 4: Apply crossover and mutation to the selected parents
    offspring = apply_crossover_and_mutation(parents_list, crossover_probability=crossover,
                                                     num_points=num_points,
                                                     mutation_rate=mutation, API_key=API_key, reference=reference,
                                                     if_api=if_api)

    # Combine elites with the mutated offspring
    next_generation = elites + offspring[:batch_size-num_elites]

    assert len(next_generation) == batch_size
    return next_generation


### GA ###
def roulette_wheel_selection(data_list, score_list, num_selected, if_softmax=True):
    if if_softmax:
        selection_probs = np.exp(score_list - np.max(score_list))
        selection_probs = selection_probs / selection_probs.sum()
    else:
        total_score = sum(score_list)
        selection_probs = [score / total_score for score in score_list]

    selected_indices = np.random.choice(len(data_list), size=num_selected, p=selection_probs, replace=True)

    selected_data = [data_list[i] for i in selected_indices]
    return selected_data


def apply_crossover_and_mutation(selected_data, crossover_probability=0.5, num_points=3, mutation_rate=0.01,
                                 API_key=None,
                                 reference=None, if_api=True):
    offspring = []

    for i in range(0, len(selected_data), 2):
        parent1 = selected_data[i]
        parent2 = selected_data[i + 1] if (i + 1) < len(selected_data) else selected_data[0]

        if random.random() < crossover_probability:
            child1, child2 = crossover(parent1, parent2, num_points)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)

    mutated_offspring = apply_gpt_mutation(offspring, mutation_rate, API_key, reference, if_api)

    return mutated_offspring


def crossover(str1, str2, num_points):
    # Function to split text into paragraphs and then into sentences
    def split_into_paragraphs_and_sentences(text):
        paragraphs = text.split('\n\n')
        return [re.split('(?<=[,.!?])\s+', paragraph) for paragraph in paragraphs]

    paragraphs1 = split_into_paragraphs_and_sentences(str1)
    paragraphs2 = split_into_paragraphs_and_sentences(str2)

    new_paragraphs1, new_paragraphs2 = [], []

    for para1, para2 in zip(paragraphs1, paragraphs2):
        max_swaps = min(len(para1), len(para2)) - 1
        num_swaps = min(num_points, max_swaps)

        swap_indices = sorted(random.sample(range(1, max_swaps + 1), num_swaps))

        new_para1, new_para2 = [], []
        last_swap = 0
        for swap in swap_indices:
            if random.choice([True, False]):
                new_para1.extend(para1[last_swap:swap])
                new_para2.extend(para2[last_swap:swap])
            else:
                new_para1.extend(para2[last_swap:swap])
                new_para2.extend(para1[last_swap:swap])
            last_swap = swap

        if random.choice([True, False]):
            new_para1.extend(para1[last_swap:])
            new_para2.extend(para2[last_swap:])
        else:
            new_para1.extend(para2[last_swap:])
            new_para2.extend(para1[last_swap:])

        new_paragraphs1.append(' '.join(new_para1))
        new_paragraphs2.append(' '.join(new_para2))

    return '\n\n'.join(new_paragraphs1), '\n\n'.join(new_paragraphs2)

def gpt_mutate(sentence, API_key=None):
    #print(API_key)
    openai.api_key = API_key
    system_msg = 'You are a helpful and creative assistant who writes well.'
    user_message = f'Please revise the following sentence with no changes to its length and only output the revised version, the sentences are: \n "{sentence}".\nPlease give me your revision directly without any explanation. Remember keep the original paragraph structure. Do not change the words "[REPLACE]", "[PROMPT]", "[KEEPER]", and "[MODEL]", if they are in the sentences.'
    revised_sentence = sentence
    received = False
    while not received:
        try:
            response = openai.ChatCompletion.create(model="gpt-4o-mini",
                                                    messages=[{"role": "system", "content": system_msg},
                                                              {"role": "user", "content": user_message}],
                                                    temperature=1, top_p=0.9)
            revised_sentence = response["choices"][0]["message"]["content"].replace('\n', '')
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:  # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError, Prompt error.")
                return None
            if error == AssertionError:
                print("Assert error:", sys.exc_info()[1])  # assert False
            else:
                print("API error:", error)
            time.sleep(1)
    if revised_sentence.startswith("'") or revised_sentence.startswith('"'):
        revised_sentence = revised_sentence[1:]
    if revised_sentence.endswith("'") or revised_sentence.endswith('"'):
        revised_sentence = revised_sentence[:-1]
    if revised_sentence.endswith("'.") or revised_sentence.endswith('".'):
        revised_sentence = revised_sentence[:-2]
    #print(f'revised: {revised_sentence}')
    return revised_sentence

def apply_gpt_mutation(offspring, mutation_rate=0.01, API_key=None, reference=None, if_api=True):
    if if_api:
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                if API_key is None:
                    offspring[i] = random.choice(reference[len(offspring):])
                else:
                    offspring[i] = gpt_mutate(offspring[i], API_key)
    else:
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = replace_with_synonyms(offspring[i])
    return offspring


def apply_init_gpt_mutation(offspring, mutation_rate=0.01, API_key=None, if_api=True):
    for i in tqdm(range(len(offspring)), desc='initializing...'):
        if if_api:
            if random.random() < mutation_rate:
                offspring[i] = gpt_mutate(offspring[i], API_key)
        else:
            if random.random() < mutation_rate:
                offspring[i] = replace_with_synonyms(offspring[i])
    return offspring


def replace_with_synonyms(sentence, num=10):
    T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
         "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(sentence)
    uncommon_words = [word for word in words if word.lower() not in stop_words and word.lower() not in T]
    selected_words = random.sample(uncommon_words, min(num, len(uncommon_words)))
    for word in selected_words:
        synonyms = wordnet.synsets(word)
        if synonyms and synonyms[0].lemmas():
            synonym = synonyms[0].lemmas()[0].name()
            sentence = sentence.replace(word, synonym, 1)
    #print(f'revised: {sentence}')
    return sentence


### HGA ###
def autodan_sample_control_hga(word_dict, control_suffixs, score_list, num_elites, batch_size, crossover=0.5,
                               mutation=0.01, API_key=None, reference=None, if_api=True):
    score_list = [-x for x in score_list]
    # Step 1: Sort the score_list and get corresponding control_suffixs
    sorted_indices = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
    sorted_control_suffixs = [control_suffixs[i] for i in sorted_indices]

    # Step 2: Select the elites
    elites = sorted_control_suffixs[:num_elites]
    parents_list = sorted_control_suffixs[num_elites:]

    # Step 3: Construct word list
    word_dict = construct_momentum_word_dict(word_dict, control_suffixs, score_list)
    print(f"Length of current word dictionary: {len(word_dict)}")

    # check the length of parents
    parents_list = [x for x in parents_list if len(x) > 0]
    if len(parents_list) < batch_size - num_elites:
        print("Not enough parents, using reference instead.")
        parents_list += random.choices(reference[batch_size:], k = batch_size - num_elites - len(parents_list))
        
    # Step 4: Apply word replacement with roulette wheel selection
    offspring = apply_word_replacement(word_dict, parents_list, crossover)
    offspring = apply_gpt_mutation(offspring, mutation, API_key, reference, if_api)

    # Combine elites with the mutated offspring
    next_generation = elites + offspring[:batch_size-num_elites]

    assert len(next_generation) == batch_size
    return next_generation, word_dict

def construct_momentum_word_dict(word_dict, control_suffixs, score_list, topk=-1):
    T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
         "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
    stop_words = set(stopwords.words('english'))
    if len(control_suffixs) != len(score_list):
        raise ValueError("control_suffixs and score_list must have the same length.")

    word_scores = defaultdict(list)

    for prefix, score in zip(control_suffixs, score_list):
        words = set(
            [word for word in nltk.word_tokenize(prefix) if word.lower() not in stop_words and word.lower() not in T])
        for word in words:
            word_scores[word].append(score)

    for word, scores in word_scores.items():
        avg_score = sum(scores) / len(scores)
        if word in word_dict:
            word_dict[word] = (word_dict[word] + avg_score) / 2
        else:
            word_dict[word] = avg_score

    sorted_word_dict = OrderedDict(sorted(word_dict.items(), key=lambda x: x[1], reverse=True))
    if topk == -1:
        topk_word_dict = dict(list(sorted_word_dict.items()))
    else:
        topk_word_dict = dict(list(sorted_word_dict.items())[:topk])
    return topk_word_dict


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def word_roulette_wheel_selection(word, word_scores):
    if not word_scores:
        return word
    min_score = min(word_scores.values())
    adjusted_scores = {k: v - min_score for k, v in word_scores.items()}
    total_score = sum(adjusted_scores.values())
    pick = random.uniform(0, total_score)
    current_score = 0
    for synonym, score in adjusted_scores.items():
        current_score += score
        if current_score > pick:
            if word.istitle():
                return synonym.title()
            else:
                return synonym

def replace_with_best_synonym(sentence, word_dict, crossover_probability):
    stop_words = set(stopwords.words('english'))
    T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
         "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
    paragraphs = sentence.split('\n\n')
    modified_paragraphs = []
    min_value = min(word_dict.values())
    #print(sentence)

    for paragraph in paragraphs:
        words = replace_quotes(nltk.word_tokenize(paragraph))
        count = 0
        for i, word in enumerate(words):
            if random.random() < crossover_probability:
                if word.lower() not in stop_words and word.lower() not in T:
                    synonyms = get_synonyms(word.lower())
                    word_scores = {syn: word_dict.get(syn, min_value) for syn in synonyms}
                    best_synonym = word_roulette_wheel_selection(word, word_scores)
                    if best_synonym:
                        words[i] = best_synonym
                        count += 1
                        if count >= 5:
                            break
            else:
                if word.lower() not in stop_words and word.lower() not in T:
                    synonyms = get_synonyms(word.lower())
                    word_scores = {syn: word_dict.get(syn, 0) for syn in synonyms}
                    best_synonym = word_roulette_wheel_selection(word, word_scores)
                    if best_synonym:
                        words[i] = best_synonym
                        count += 1
                        if count >= 5:
                            break
        modified_paragraphs.append(join_words_with_punctuation(words))
    return '\n\n'.join(modified_paragraphs)

def replace_quotes(words):
    new_words = []
    quote_flag = True

    for word in words:
        if word in ["``", "''"]:
            if quote_flag:
                new_words.append('“')
                quote_flag = False
            else:
                new_words.append('”')
                quote_flag = True
        else:
            new_words.append(word)
    return new_words

def apply_word_replacement(word_dict, parents_list, crossover=0.5):
    return [replace_with_best_synonym(sentence, word_dict, crossover) for sentence in parents_list]

def join_words_with_punctuation(words):
    sentence = words[0]
    previous_word = words[0]
    flag = 1
    for word in words[1:]:
        if word in [",", ".", "!", "?", ":", ";", ")", "]", "}", '”']:
            sentence += word
        else:
            if previous_word in ["[", "(", "'", '"', '“']:
                if previous_word in ["'", '"'] and flag == 1:
                    sentence += " " + word
                else:
                    sentence += word
            else:
                if word in ["'", '"'] and flag == 1:
                    flag = 1 - flag
                    sentence += " " + word
                elif word in ["'", '"'] and flag == 0:
                    flag = 1 - flag
                    sentence += word
                else:
                    if "'" in word and re.search('[a-zA-Z]', word):
                        sentence += word
                    else:
                        sentence += " " + word
        previous_word = word
    return sentence


class AutoDanSuffixManager:
    """
    自动化对抗攻击后缀管理器，用于构建输入prompt和计算各部分token位置(slice)。
    
    功能：
    - 维护对话模板和用户输入。
    - 构建完整prompt。
    - 提供用于计算loss的token切片位置。

    参数:
        tokenizer: 分词器对象。
        conv_template: 对话模板对象，需提供 roles, append_message(), get_prompt(), update_last_message() 等方法。
        instruction: 用户输入的自然语言指令。
        target: 模型的目标回答文本。
        adv_string: 对抗后缀（adversarial suffix）。

    属性:
        _user_role_slice, _goal_slice, _control_slice,
        _assistant_role_slice, _target_slice, _loss_slice:
        这些切片定义了prompt中不同角色和内容的token索引范围。
    """
    
    ANSWER_SUFFIX = " Please provide your answer based on the context above."

    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string + self.ANSWER_SUFFIX

    def get_prompt(self, adv_string=None):
        """
        构建完整prompt，并计算token位置切片。

        参数:
            adv_string: 可选的替换后缀。如果提供，会替换默认的self.adv_string。

        返回:
            prompt: 完整的输入字符串
        """
        if adv_string is not None:
            # 替换标记
            self.adv_string = adv_string.replace('[REPLACE]', self.instruction.lower())

        # 添加用户和助手消息
        self.conv_template.append_message(self.conv_template.roles[0], self.adv_string)
        self.conv_template.append_message(self.conv_template.roles[1], self.target)
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)

        # === llama-2 模板处理逻辑 ===
        if self.conv_template.name == 'llama-2':
            self._build_slices_llama2()
        else:
            self._build_slices_generic(prompt, encoding)

        # 重置messages，防止污染状态
        self.conv_template.messages = []
        return prompt

    def _build_slices_llama2(self):
        """构建 llama-2 模型专用的token切片"""
        self.conv_template.messages = []
        self.conv_template.append_message(self.conv_template.roles[0], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._user_role_slice = slice(None, len(toks))

        # goal slice
        self.conv_template.update_last_message(self.adv_string)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))
        self._control_slice = self._goal_slice

        # assistant slice
        self.conv_template.append_message(self.conv_template.roles[1], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

        # target slice
        self.conv_template.update_last_message(self.target)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
        self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

    def _build_slices_generic(self, prompt, encoding):
        """构建通用模型的token切片"""
        python_tokenizer = self.conv_template.name == 'oasst_pythia'
        try:
            encoding.char_to_token(len(prompt) - 1)
        except Exception:
            python_tokenizer = True

        if python_tokenizer:
            self._build_slices_python_tokenizer()
        else:
            self._build_slices_standard(prompt, encoding)

    def _build_slices_python_tokenizer(self):
        """针对python_tokenizer的特殊处理"""
        self.conv_template.messages = []
        self.conv_template.append_message(self.conv_template.roles[0], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._user_role_slice = slice(None, len(toks))

        # goal slice
        self.conv_template.update_last_message(self.adv_string)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks) - 1))
        self._control_slice = self._goal_slice

        # assistant slice
        self.conv_template.append_message(self.conv_template.roles[1], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

        # target slice
        self.conv_template.update_last_message(self.target)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 1)
        self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 2)

    def _build_slices_standard(self, prompt, encoding):
        """标准分词器下的slice构建逻辑"""
        self._system_slice = slice(
            None,
            encoding.char_to_token(len(self.conv_template.system))
        )
        self._user_role_slice = slice(
            encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
            encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
        )
        self._goal_slice = slice(
            encoding.char_to_token(prompt.find(self.adv_string)),
            encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
        )
        self._control_slice = self._goal_slice
        self._assistant_role_slice = slice(
            encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
            encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
        )
        self._target_slice = slice(
            encoding.char_to_token(prompt.find(self.target)),
            encoding.char_to_token(prompt.find(self.target) + len(self.target))
        )
        self._loss_slice = slice(
            encoding.char_to_token(prompt.find(self.target)) - 1,
            encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
        )

    def get_input_ids(self, adv_string=None):
        """
        根据prompt生成input_ids。

        参数:
            adv_string: 可选的新后缀
            model: 模型对象（仅为接口一致性）

        返回:
            input_ids: torch.Tensor，直到target_slice结束的输入序列
        """
        suffix = adv_string + self.ANSWER_SUFFIX if adv_string else self.adv_string
        prompt = self.get_prompt(adv_string=suffix)
        toks = self.tokenizer(prompt).input_ids
        return torch.tensor(toks[:self._target_slice.stop])


def remove_after_pattern(lst, pattern):
    pattern_length = len(pattern)
    for i in range(len(lst) - pattern_length + 1):
        if lst[i:i + pattern_length].tolist() == pattern.tolist():
            return lst[:i + pattern_length]
    return lst

class AutoDAN():
    
    def __init__(self) -> None:
        pass

    def compute_loss(
        self, 
        tokenizer,
        conv_template,
        instruction,
        target,
        model,
        device,
        test_controls,
        crit,
        dis=None
    ):
        """
        计算一组候选攻击后缀的loss。
        
        参数:
            tokenizer: 分词器
            conv_template: 对话模板
            instruction: 用户指令
            target: 攻击目标文本
            model: 待攻击的模型
            device: 设备 (cuda/cpu)
            test_controls: List[str] 候选后缀
            crit: 损失函数 (criterion)
            dis: (可选) 分布约束或额外loss输入
            
        返回:
            losses: torch.Tensor, 每个后缀的loss值
        """
        input_ids_list = []
        target_slices = []

        # 遍历每个候选后缀
        for adv_suffix in test_controls:
            # 1. 构造SuffixManager获取输入token
            suffix_manager = AutoDanSuffixManager(
                tokenizer=tokenizer,
                conv_template=conv_template,
                instruction=instruction,
                target=target,
                adv_string=adv_suffix,
            )
            input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)
            
            # import pdb
            # pdb.set_trace()

            # === Token剪枝：去掉部分低排名token ===
            torch.save(torch.tensor([]), "indices4.pt")
            # model.prepare_inputs_for_generation(input_ids=None)

            marker = tokenizer("[/INST]", return_tensors="pt").input_ids[0][1:5]
            marker = marker.to(device).cpu()

            # 删除marker之后的token
            trimmed_ids = remove_after_pattern(input_ids.cpu(), marker)
            # cache_outputs = model(
            #     input_ids=trimmed_ids.unsqueeze(0).to(device),
            #     past_key_values=None,
            #     use_cache=True,
            # )

            ranked_indices = rank_indices(torch.load("indices4.pt").cpu().numpy())

            # 取前20%的token index进行删除
            num_tokens_to_remove = int(len(trimmed_ids) * 0.2)
            eviction_index = [idx for idx, _ in ranked_indices[:num_tokens_to_remove]]

            # 构建新输入序列
            filtered_ids = [
                tok_id for idx, tok_id in enumerate(input_ids) if idx not in eviction_index
            ]
            input_ids = torch.tensor(filtered_ids).to(device)
            input_ids_list.append(input_ids)

            # 修正target slice（考虑删除token后的offset）
            adjusted_start = suffix_manager._target_slice.start - num_tokens_to_remove
            adjusted_stop = suffix_manager._target_slice.stop - num_tokens_to_remove
            target_slices.append(slice(adjusted_start, adjusted_stop))
            
        # import pdb
        # pdb.set_trace()

        # === Padding输入序列，组成batch ===
        pad_token_id = 0
        # 避免pad_token_id出现在序列中
        while any(pad_token_id in ids for ids in input_ids_list):
            pad_token_id += 1

        max_length = max(ids.size(0) for ids in input_ids_list)
        padded_input_ids_list = [
            torch.cat([ids, torch.full((max_length - ids.size(0),), pad_token_id, device=device)])
            for ids in input_ids_list
        ]
        input_ids_tensor = torch.stack(padded_input_ids_list, dim=0)
        attn_mask = (input_ids_tensor != pad_token_id).type(input_ids_tensor.dtype)

        # === 前向传播计算logits ===
        model.prepare_inputs_for_generation(input_ids=input_ids_tensor)
        logits = forward(
            model=model,
            input_ids=input_ids_tensor,
            attention_mask=attn_mask,
            batch_size=len(test_controls),
            target=target_slices,
        )

        # === 计算每个序列的loss ===
        losses = []
        for idx, target_slice in enumerate(target_slices):
            loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
            logits_slice = logits[idx, loss_slice, :].unsqueeze(0).transpose(1, 2)
            targets = input_ids_tensor[idx, target_slice].unsqueeze(0)

            if dis is not None:
                # 如果使用了dis分布约束
                logits_cal = logits_slice.unsqueeze(0).expand_as(dis)
                loss = -crit(logits_cal, dis).mean()
            else:
                loss = crit(logits_slice, targets)

            losses.append(loss)

        # 清理显存
        del input_ids_list, target_slices, input_ids_tensor, attn_mask
        gc.collect()

        return torch.stack(losses)
