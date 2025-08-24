import torch


class autodan_SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string + " Please provide your answer based on the context above."

    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string.replace('[REPLACE]', self.instruction.lower())

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids

        if self.conv_template.name == 'llama-2':
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.adv_string}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))
            self._control_slice = self._goal_slice

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
            self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)

        else:
            python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
            try:
                encoding.char_to_token(len(prompt) - 1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.adv_string}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks) - 1))
                self._control_slice = self._goal_slice

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 1)
                self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 2)
            else:
                self._system_slice = slice(
                    None,
                    encoding.char_to_token(len(self.conv_template.system))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(self.adv_string)),
                    encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
                )
                self._control_slice = self._goal_slice
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(
                        prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
                )

        self.conv_template.messages = []

        return prompt

    

    def get_input_ids(self, adv_string=None, model = None):
        prompt = self.get_prompt(adv_string=adv_string+" Please provide your answer based on the context above.")
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])
        

        return input_ids


def get_score_autodan(tokenizer, conv_template, instruction, target, model, device, test_controls=None, crit=None, dis=None):
    # Convert all test_controls to token ids and find the max length
    input_ids_list = []
    target_slices = []
    for item in test_controls:
        suffix_manager = autodan_SuffixManager(
            tokenizer=tokenizer,
            conv_template=conv_template,
            instruction=instruction,
            target=target,
            adv_string=item,
        )
        input_ids = suffix_manager.get_input_ids(adv_string=item,model=model).to(device)
        #print(tokenizer.decode(input_ids,skip_special_tokens=True,clean_up_tokenization_spaces=True,spaces_between_special_tokens=False,))
        #print(input_ids.size())
        empty = torch.tensor([])
        torch.save(empty,"indices4.pt")
        model.prepare_inputs_for_generation(input_ids=None)
        marker = tokenizer("[/INST]",return_tensors="pt").input_ids[0][1:5]
        my_ids = remove_after_pattern(input_ids.cpu(), marker.cpu())
        cache_outputs = model(
            input_ids= my_ids.unsqueeze(0).to(device),
            past_key_values=None,
            use_cache=True,
            )
        my_indices = torch.load("indices4.pt")
        ranked_indices = rank_indices(my_indices.cpu().numpy())
        eviction_index = []
        mycount = 0
        #print(int(len(my_ids)*0.2))
        for element in ranked_indices:
            if mycount<int(len(my_ids)*0.2):
                eviction_index.append(element[0])
            mycount += 1
        #print(eviction_index)
        new_input_ids = []
        for i in range (len(input_ids)):
            if i not in eviction_index:
                new_input_ids.append(input_ids[i])
        input_ids = torch.tensor(new_input_ids).to(device)
        #print(tokenizer.decode(input_ids,skip_special_tokens=True,clean_up_tokenization_spaces=True,spaces_between_special_tokens=False,))
        input_ids_list.append(input_ids)
        #target_slices.append(suffix_manager._target_slice)
        target_slices.append(slice(suffix_manager._target_slice.start-int(len(my_ids)*0.2),suffix_manager._target_slice.stop-int(len(my_ids)*0.2)))
        #print(tokenizer.decode(input_ids[slice(suffix_manager._target_slice.start-20,suffix_manager._target_slice.stop-20)],skip_special_tokens=True,clean_up_tokenization_spaces=True,spaces_between_special_tokens=False,))
    

    # Pad all token ids to the max length
    pad_tok = 0
    for ids in input_ids_list:
        while pad_tok in ids:
            pad_tok += 1

    # Find the maximum length of input_ids in the list
    max_input_length = max([ids.size(0) for ids in input_ids_list])

    # Pad each input_ids tensor to the maximum length

    padded_input_ids_list = []
    for ids in input_ids_list:
        pad_length = max_input_length - ids.size(0)
        padded_ids = torch.cat([ids, torch.full((pad_length,), pad_tok, device=device)], dim=0)
        #print(ids.size(), padded_ids.size())
        padded_input_ids_list.append(padded_ids)


    # Stack the padded input_ids tensors
    input_ids_tensor = torch.stack(padded_input_ids_list, dim=0)

    #print(input_ids_tensor.size(),)
    attn_mask = (input_ids_tensor != pad_tok).type(input_ids_tensor.dtype)
    model.prepare_inputs_for_generation(input_ids=input_ids_tensor)

    # Forward pass and compute loss
    logits = forward(model=model, input_ids=input_ids_tensor, attention_mask=attn_mask, batch_size=len(test_controls), target = target_slices)
    #print(logits.size())
    losses = []
    for idx, target_slice in enumerate(target_slices):
        loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
        #print(tokenizer.decode(input_ids_tensor[idx, loss_slice]))
        logits_slice = logits[idx, loss_slice, :].unsqueeze(0).transpose(1, 2)
        targets = input_ids_tensor[idx, target_slice].unsqueeze(0)
        #new_target = tokenizer("cannot",return_tensors="pt").input_ids[0][1]
        #targets = new_target.repeat(target_slice.stop-target_slice.start).unsqueeze(0).to(logits.device)
        if dis is not None:
            logits_cal = logits_slice.unsqueeze(0).expand_as(dis)
            loss = -crit(logits_cal, dis).mean()
            losses.append(loss)
        else:
            loss = crit(logits_slice, targets)
            #print(targets)
            losses.append(loss)

    del input_ids_list, target_slices, input_ids_tensor, attn_mask
    gc.collect()
    return torch.stack(losses)

def remove_after_pattern(lst, pattern):
    pattern_length = len(pattern)
    for i in range(len(lst) - pattern_length + 1):
        if lst[i:i + pattern_length].tolist() == pattern.tolist():
            return lst[:i + pattern_length]
    return lst
