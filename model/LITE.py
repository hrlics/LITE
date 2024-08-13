import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from datasets import load_from_disk
from utils import replace_mask_with_values
from transformers import DistilBertForMaskedLM, AutoTokenizer, SwinModel, DistilBertModel, LlamaModel, LlamaConfig, LlamaTokenizer


'''LLM Decoder'''
class LLM_Decoder(nn.Module):

    def __init__(self, llm_layers=6, d_model=768, llama_token=None):
        super(LLM_Decoder, self).__init__()
        self.d_model = 768 # dimension of semantic time-series encoder and vision encoder
        d_llm = 4096 # dimension of LLM
        token = llama_token
        assert token, 'Need token for LLaMA2!'
        llm_path = 'meta-llama/Llama-2-7b-hf'
        
        self.in_layer = nn.Linear(d_model, d_llm)
        self.out_layer = nn.Sequential(
            nn.Linear(d_llm, 768),
            nn.Dropout(0.3),
            nn.Linear(768, 1)
        )

        self.llama_config = LlamaConfig.from_pretrained(llm_path, token=token)
        self.llama_config.num_hidden_layers = llm_layers
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True
        self.llama = LlamaModel.from_pretrained(
            llm_path,
            token=token,
            trust_remote_code=True,
            local_files_only=False,
            config=self.llama_config,
            load_in_4bit=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            token=token,
            trust_remote_code=True,
            local_files_only=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # freeze the LLM
        for param in self.llama.parameters():
            param.requires_grad = False


    def forward(self, text_embed, vision_embed, stats):
        text_embed = self.in_layer(text_embed) 
        vision_embed = self.in_layer(vision_embed)

        B = text_embed.shape[0]
        min_values = torch.zeros(B)
        max_values = torch.zeros(B)
        medians = torch.zeros(B)
        trends = torch.zeros(B)
        
        for i in range(B):
            valid_stats = stats[i][stats[i] != -11.] # -11. represents invalid for CRW-Temp and CRW-Flow
            # valid_stats = stats[i][~torch.isnan(stats[i])] # nan represents invalid for AGR

            if len(valid_stats) > 0:
                min_values[i] = torch.min(valid_stats)
                max_values[i] = torch.max(valid_stats)
                medians[i] = torch.median(valid_stats)

                if len(valid_stats) > 1:
                    trends[i] = valid_stats.diff().sum()
                else: # no valid stats
                    trends[i] = 0
            else:
                min_values[i], max_values[i], medians[i], trends[i] = torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan')), 0

        # incorporate prompt
        prompt = []
        for b in range(text_embed.shape[0]):
            min_values_str = str(min_values[b])
            max_values_str = str(max_values[b])
            median_values_str = str(medians[b])

            if trends[b] > 0:
                trend= 'upward'
            elif trends[b] < 0:
                trend= 'downward'
            else:
                trend= 'no observation'

            prompt_ = (
                f"<|start_prompt|>Dataset description: The Christina River Watershed Temperature (CRW-Temp) is a dataset containing water temperature observations from 42 river segments."
                f"Task description: predict the water temperature given the observed meteorological features represented in the image and text spaces; "
                # f"<|start_prompt|>Dataset description: The Christina River Watershed Flow (CRW-Flow) is a dataset containing water flow observations from 42 river segments."
                # f"Task description: predict the water flow given the observed meteorological features represented in the image and text spaces; "
                # f"<|start_prompt|>Dataset description: The Agriculture Nitrous Oxide (AGR) is a dataset containing agricultural nitrou emission observations from 6 chambers."
                # f"Task description: predict the nitrou emission given the observed meteorological features represented in the image and text spaces; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {trend} <|<end_prompt>|>"
            )
            prompt.append(prompt_)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids    
        prompt_embeds = self.llama.get_input_embeddings()(prompt.to('cuda'))  # (batch, prompt_token, dim)
        llama_enc_out = torch.cat([prompt_embeds, text_embed, vision_embed], dim=1)

        # decode
        dec_out = self.llama(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out=torch.mean(dec_out, dim=1)
        dec_out=self.out_layer(dec_out) # (btz, 1)
        return dec_out


'''SMoE for imputation'''
class MLP(nn.Module):
    def __init__(self, d_model, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.fc3 = nn.Linear(d_model, 1)
    
    def forward(self, x):
        out=self.dropout(self.relu(self.fc1(x)))
        out=self.fc2(out)
        out=x+out
        out=self.fc3(out) # z: [btz, 1]
        return out 

class MoE_ffn(nn.Module):

    def __init__(self, num_experts=8, hidden_dim=768, dropout_rate=0.3, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_experts=num_experts
        self.d_model=hidden_dim
        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.experts=self.experts = nn.ModuleList(
            [  MLP(hidden_dim, dropout_rate=dropout_rate) for _ in range(self.num_experts)]
        )
               
    def forward(self, hidden_states):
        num_masks, d_model = hidden_states.shape
        router_logits = self.gate(hidden_states) # num_masks, num_experts
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float) # num_masks, num_experts
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) #(num_masks, num_experts), (num_masks, num_experts)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True) # normalization
        routing_weights = routing_weights.to(hidden_states.dtype) #(num_masks, topk)
        final_outputs= torch.zeros((num_masks, 1), device=hidden_states.device)

        for i, expert_indices in enumerate(selected_experts):
            for j, w in enumerate(expert_indices):
                final_outputs[i,:] += routing_weights[i, j] * self.experts[j](hidden_states[i])

        return final_outputs


'''The LITE Model'''
class Model(nn.Module):

    def __init__(self, bert_path=None, vison_path=None, d_model=768, dropout_rate=0.3, num_experts=8, topk=2, llm_layers=6,llama_token=None, device=None):
        super(Model, self).__init__()
        self.device = device
        
        # language encoder
        self.distilbert = DistilBertForMaskedLM.from_pretrained(bert_path) 
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path) 
        # vision encoder
        self.vision = SwinModel.from_pretrained(vison_path)

        # SMoE for imputation
        self.moe_layer = MoE_ffn(num_experts=num_experts, hidden_dim=d_model, dropout_rate=dropout_rate, top_k=topk)
        # frozen LLM decoder
        self.decoder = LLM_Decoder(d_model=d_model, llm_layers=llm_layers, llama_token=llama_token) 
    

    def impute(self, hidden_states, input_ids):
        sentence_ids, mask_token_ids = torch.where(input_ids == self.tokenizer.mask_token_id)

        if len(mask_token_ids)==0: # if no missing feature
            return torch.ones(hidden_states.shape[0], 768).to(self.device), (torch.ones(hidden_states.shape[0], 5)*-10.).to(self.device)
        
        gen_sentences = []
        og_mask_sentences = [self.tokenizer.decode(sequence[1:-1]) for sequence in input_ids] # why 1:-1? -> remove <cls> & <sep>
        
        total_impute_val = []
        for id, sentence in enumerate(hidden_states):
            mask_ids = mask_token_ids[sentence_ids == id] 
            mask_embeds = sentence[mask_ids] # z: (num_of_mask, d_encoder)
            
            # Use SMoE to impute
            impute_val = self.moe_layer(mask_embeds) # z: (num_of_mask, 1)
            tmp = torch.ones(4)*-10.
            for idx, val in enumerate(impute_val):
                tmp[idx] = val
            total_impute_val.append(tmp)
            
            tmp_sentence = og_mask_sentences[id]
            gen_sentences.append(replace_mask_with_values(tmp_sentence, impute_val))
        
        total_impute_val = torch.stack(total_impute_val, dim=0).to(self.device)
        
        # replace '[mask]' with imputed value 
        new_inputs = self.tokenizer(gen_sentences, return_tensors='pt', padding=True).to(self.device)
        last_hs_imputed = self.distilbert.base_model(**new_inputs).last_hidden_state # (btz, 1, 768)
        last_hs_imputed = last_hs_imputed[:, 0, :]
        return last_hs_imputed, total_impute_val


    def forward(self, pixel_values, input_ids, attention_mask, week_sentences, month_sentences, stats):
        
        # week data
        week_s_inputs = [
            self.tokenizer(week_sentence, truncation=True, padding=True, return_tensors="pt").to(self.device) 
            for week_sentence in week_sentences ] # z: [btz, 6, seq_len]
        last_hs_week = [
            self.distilbert.base_model(**week_s_input).last_hidden_state[:, 0, :] # [cls] token
            for week_s_input in week_s_inputs] 
        last_hs_week= torch.stack(last_hs_week, dim=0) # z: [btz, 6, 768]

        # month data
        month_s_inputs = [
            self.tokenizer(month_sentence,truncation=True, padding=True, return_tensors="pt").to(self.device) 
            for month_sentence in month_sentences] # z: [btz, 11, seq_len]
        last_hs_month = [
            self.distilbert.base_model(**month_s_input).last_hidden_state[:, 0, :] # [cls] token
            for month_s_input in month_s_inputs] # z: [btz, 11]
        last_hs_month = torch.stack(last_hs_month, dim=0) # (btz, 11, 768)
        
        # SMoE imputation
        last_hs = self.distilbert.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # (btz, seq_len, 768)        
        imputed_text_embeds, impute_val = self.impute(last_hs, input_ids)
        imputed_text_embeds = imputed_text_embeds.unsqueeze(1) # (btz, 1, 768)

        # fuse multimodal representation
        text_embeds=torch.cat([
            imputed_text_embeds, last_hs_week, last_hs_month], dim=1) # (btz, 1 [today] + 6 [this week] + 11 [the same day on every month on this year], 768)
        vision_embed = self.vision(pixel_values=pixel_values).last_hidden_state[:, 0, :] # (btz, 768)
        vision_embed=vision_embed.unsqueeze(1) # (btz, 1, 768)
        outputs = self.decoder(text_embeds, vision_embed, stats)

        return outputs, impute_val






