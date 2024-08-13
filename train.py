import os
import random
import wandb
import math
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error 

import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoTokenizer, DistilBertForMaskedLM, AdamW, DataCollatorWithPadding, AutoFeatureExtractor
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision.transforms import Compose, ToTensor
# for Mixed-precision Training
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast

from models.LITE import Model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', type=str, default='datasets/CRW')
parser.add_argument('--dataset_name', type=str, default='CRW', help='name used to save checkpoints')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--train_batchsize', type=int, default=16)
parser.add_argument('--test_batchsize', type=int, default=32) 
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--bert_path', type=str, default='distilbert/distilbert-base-uncased')
parser.add_argument('--vision_path', type=str, default='microsoft/swin-tiny-patch4-window7-224')
parser.add_argument('--d_model', type=int, default=768, help='hidden dimension of model')
parser.add_argument('--SEED', type=int, default=3407)
parser.add_argument('--dropout_rate', type=float, default=0.3, help='drop out rate of MLP in the model')
parser.add_argument('--warm_up_epochs', type=int, default=5)
parser.add_argument('--wandb', type=bool, default=False)
parser.add_argument('--llama_token', type=str, default=None)

args, _ = parser.parse_known_args()
dataset_folder=args.dataset_folder
dataset_name=args.dataset_name
num_epochs=args.epochs
learning_rate=args.learning_rate
train_btz=args.train_batchsize
test_btz=args.test_batchsize
bert_path=args.bert_path
vision_path=args.vision_path
d_model=args.d_model
dropout_rate=args.dropout_rate
llama_token=args.llama_token
SEED=args.SEED

assert llama_token, 'Need token for LLaMA2!'


print(dataset_folder)
print(bert_path,'\n')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(SEED)

checkpoint = bert_path
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
feature_extractor = AutoFeatureExtractor.from_pretrained(vision_path)
transforms= Compose([ToTensor()])

def preprocess(examples):
    examples["pixel_values"] = [
        transforms(image.convert("RGB").resize((224,224))) for image in examples["image"]]
    text_inputs = tokenizer(examples["sentence"], truncation=True, padding=True, max_length=512, return_tensors="pt")
    for key, value in text_inputs.items():
        examples[key] = value
    return examples


checkpoint_path=f'checkpoints/{dataset_name}'
os.makedirs(checkpoint_path, exist_ok=True)
print(f"save the model checkpoints at:{checkpoint_path}.\n")
print("start training")

# train dataset
train_dataset=load_from_disk(f"{dataset_folder}/{dataset_name}_dataset_withImage_train")
train_dataset = train_dataset.map(preprocess, batched=True, batch_size=256)
if dataset_name=="CRW" or dataset_name=='Flow':
    train_dataset = train_dataset.remove_columns(['sentence', 'seg_idx','date_idx','image'])
elif dataset_name=="Agri":
    train_dataset = train_dataset.remove_columns(['sentence', 'chamber_idx','hour_idx','image'])
train_dataset.set_format("torch")

# test dataset
test_dataset=load_from_disk(f"{dataset_folder}/{dataset_name}_dataset_withImage_test")
test_dataset = test_dataset.map(preprocess, batched=True, batch_size=256)
if dataset_name=="CRW" or dataset_name=='Flow':
    test_dataset = test_dataset.remove_columns(['sentence', 'seg_idx','date_idx','image'])
elif dataset_name=="Agri":
    test_dataset = test_dataset.remove_columns(['sentence', 'chamber_idx','hour_idx','image'])
test_dataset.set_format("torch")

print(f"train len: {len(train_dataset)}, test len: {len(test_dataset)}")

# dataloader
collate_fn=DataCollatorWithPadding(tokenizer, padding=True)
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=train_btz)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=test_btz)
train_loader = tqdm(train_loader, dynamic_ncols=True)
test_loader = tqdm(test_loader , dynamic_ncols=True)


device='cuda'
model=Model(bert_path, vision_path, d_model, dropout_rate, num_experts=8, topk=2, llm_layers=6, llama_token=llama_token, device=device)
model=model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
# warm_up_with_cosine_lr
warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * ( math.cos((epoch - args.warm_up_epochs) /(args.epochs - args.warm_up_epochs) * math.pi) + 1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

criterion_regress = RMSELoss()
criterion_impute = RMSELoss()
# mixed-precision training
scaler = GradScaler()

if args.wandb:
    wandb.init(project="LITE", name=f'{dataset_name}_btz{train_btz}_lr{learning_rate}_dropout{dropout_rate}_warmup{args.warm_up_epochs}')
lowest_regression_loss=1000.0

for epoch in range(num_epochs):

    #==== train ===
    model.train()
    mean_loss = torch.zeros(1).to(device)
    iteration=0
    
    for batch in tqdm(train_loader):
        impute_target=batch['masked_number'].flatten().to(device)
        regress_target=batch['label'].to(device)
        pixel_values=batch['pixel_values'].to(device)
        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        stats=batch['stats'].to(device)

        # multi-granularity information
        week_sentence=batch['week_sentence']
        week_sentence = [list(item) for item in zip(*week_sentence)] #z: (btz, 6)
        month_sentence=batch['month_sentence']
        month_sentence = [list(item) for item in zip(*month_sentence)] # (btz, 11)
        
        with autocast():
            outputs, impute_val = model(pixel_values, input_ids, attention_mask, week_sentence, month_sentence, stats) 
            outputs = outputs.flatten()
            impute_val = impute_val.flatten()
        
            loss_impute = criterion_impute(impute_val, impute_target)
            loss_regress = criterion_regress(outputs, regress_target)
            loss = (loss_impute + loss_regress)/2 # CRW-Temp
            mean_loss = (mean_loss * iteration + loss.detach()) / (iteration + 1)
            iteration+=1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        

    #==== test ====
    model.eval()
    mean_loss_t = 0
    with torch.no_grad():
        loss_impute_t_all=[]
        loss_regress_t_all=[]
        loss_t_all=[]
        y_test=[]
        y_predict=[]
        for batch_t in tqdm(test_loader):

            impute_target_t=batch_t['masked_number'].flatten().to(device)
            regress_target_t=batch_t['label'].to(device) #CRW
            pixel_values=batch_t['pixel_values'].to(device)
            input_ids=batch_t['input_ids'].to(device)
            attention_mask=batch_t['attention_mask'].to(device)
            stats=batch_t['stats'].to(device)

            # multi-granularity information
            week_sentence=batch_t['week_sentence']
            week_sentence = [list(item) for item in zip(*week_sentence)] # (btz, 6)
            month_sentence=batch_t['month_sentence']
            month_sentence = [list(item) for item in zip(*month_sentence)] # (btz, 11)
        
            outputs_t, impute_val_t = model(pixel_values, input_ids, attention_mask, week_sentence, month_sentence, stats)            
            outputs_t=outputs_t.flatten()
            impute_val_t=impute_val_t.flatten()

            loss_impute_t=criterion_impute(impute_val_t, impute_target_t)
            loss_regress_t=criterion_regress(outputs_t, regress_target_t)
            loss_t=( loss_impute_t + loss_regress_t )/2

            loss_impute_t_all.append(loss_impute_t.item())
            loss_regress_t_all.append(loss_regress_t.item())
            loss_t_all.append(loss_t.item())

            y_test.extend(regress_target_t.detach().cpu().numpy())
            y_predict.extend(outputs_t.detach().cpu().numpy())

        y_test = np.stack(y_test)
        y_predict = np.stack(y_predict)
        MAE = mean_absolute_error(y_test, y_predict)
        RMSE = sum(loss_regress_t_all)/len(loss_regress_t_all)
        print(f'RMSE: {RMSE:.2f}, MAE:{MAE:.2f}')
        if RMSE < lowest_regression_loss:
            lowest_regression_loss = RMSE
            print(f'current lowest regression rmse loss: {lowest_regression_loss}, MAE: {MAE}')
            torch.save(model.state_dict(), checkpoint_path + '/best_model.pt')

    # log to wandb
    wandb.log({
        "Epoch": epoch,
        "iter":iteration,
        "train_impute_loss": loss_impute.item(),
        "train_regress_loss": loss_regress.item(),
        "train_total_loss": loss.item(),
        "train_mean_loss":mean_loss.item(),
        "test_impute_loss": sum(loss_impute_t_all)/len(loss_impute_t_all),
        "test_regress_loss": sum(loss_regress_t_all)/len(loss_regress_t_all),
        "test_total_loss": sum(loss_t_all)/len(loss_t_all),
        "test_MAE": MAE,
    })
    
            
