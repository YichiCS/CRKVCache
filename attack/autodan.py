import gc

import torch

from utils import rank_indices, forward


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
            
            import pdb
            pdb.set_trace()

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
            
        import pdb
        pdb.set_trace()

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
