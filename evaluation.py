import os
import json
import logging
import evaluate
from torch.optim.optimizer import Args
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from utils.general_utils import DATASET_CLASSES, GPU_KEYS,MODEL_CLASSES,TOKENIZER_CLASSES,MODEL_PROCESSORS,DATA_PROCESSORS
#from datasets import load_metric
from evaluate import load
from rouge_score import rouge_scorer
from attrdict import AttrDict

#from torchmetrics import Accuracy
from torchmetrics.classification import Accuracy, BinaryRecall, BinaryF1Score, BinaryPrecision
#追加
from transformers import RagModel, AutoTokenizer, RagTokenizer
logger = logging.getLogger(__name__)
from transformers import (
    RagConfig
)


map_config = {
    'rag-tok-ct': RagConfig,
}


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

class Evaluation(object):
    def __init__(self, args, tokenizer, data_processor, model_processor):
        
        self.args = args
        self.tokenizer = tokenizer
        self.processor = data_processor
        self.model_processor = model_processor
        self.device = torch.device("cuda:" + str(self.args.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        self.keys_for_device = GPU_KEYS[args.model]

        self.valid_dataset = DATASET_CLASSES[self.args.model](self.args, self.tokenizer, self.processor,
                                                            self.model_processor, mode="valid")
        self.valid_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.args.valid_batch_size,
            num_workers=self.args.cpu_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self.valid_dataset.collate_fn
        )
        
        self.test_dataset = DATASET_CLASSES[self.args.model](self.args, self.tokenizer, self.processor,
                                                             self.model_processor, mode="test")
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.args.valid_batch_size,
            num_workers=self.args.cpu_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=self.test_dataset.collate_fn
        )
        self.data_map = {
            "valid": self.valid_dataloader,
            "test": self.test_dataloader,
        }
        self.dataset_map = {
            "valid": self.valid_dataset,
            "test": self.test_dataset,
        }
        if self.args.inference_path != "":
            self.inf_dataset = DATASET_CLASSES[self.args.model](self.args, self.tokenizer, self.processor,
                                                                 self.model_processor, mode="inf")
            self.inf_dataloader = DataLoader(
                self.inf_dataset,
                batch_size=self.args.valid_batch_size,
                num_workers=self.args.cpu_workers,
                shuffle=False,
                drop_last=False,
                collate_fn=self.inf_dataset.collate_fn
            )
            self.data_map["inf"] = self.inf_dataloader
            self.dataset_map["inf"] = self.inf_dataset
        
        self.bert_config = map_config[self.args.model].from_pretrained(
            self.args.backbone,
        )
        self.p_accuracy = Accuracy(num_classes=2, task="multiclass").to(self.device)
        self.k_accuracy = Accuracy(num_classes=10,task="multiclass").to(self.device)
        self.chrf_metric = evaluate.load("chrf")
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu_metric = evaluate.load("sacrebleu")
        self.f1_uni = evaluate.load("chrf")
        self.f1_p = BinaryF1Score().to(self.device)
        self.precision_p = BinaryPrecision().to(self.device)
        self.recall_p = BinaryRecall().to(self.device)


    def evaluate(self, model, epoch, typ):
        metrics = self.evaluate_rag(model, epoch, typ)
        return metrics
    
    def inference(self, model, typ):
        self.inference_rag(model, typ)

    #def evaluate_rag(self, model, epoch, typ='valid'):
    def evaluate_rag(self, model, epoch, typ):      
        self.p_accuracy.reset()
        self.k_accuracy.reset()
        self.recall_p.reset()
        self.f1_p.reset()
        self.precision_p.reset()
    
        bleu = 0
        charf = 0
        rouge1 = 0
        rouge2 = 0
        rougel = 0
        h2 = 0
        h5 = 0
        f1_persona = 0
        unif1 = 0
        
        logger.info("Starting Evaluation %s" % typ)
        model.eval()
        with torch.no_grad():
            os.makedirs(os.path.join(os.path.dirname(self.args.save_dirpath), self.args.tb_prefix),
                        exist_ok=True)
            with open(os.path.join(os.path.dirname(self.args.save_dirpath), self.args.tb_prefix,
                                   typ + f'_qualitative_results_{epoch}.json'), 'w',
                      newline='') as fw:
                tqdm_batch_iterator = tqdm(self.data_map[typ])
                qual_outputs = []
                 # 上位10件だけ処理する
                max_batches = 5

                for batch_idx, batch in enumerate(tqdm_batch_iterator):
                    if batch_idx >= max_batches:  # 制限を超えたらループを終了
                      break

                    for b_k in batch:
                        if b_k in self.keys_for_device:
                            batch[b_k] = batch[b_k].to(self.device)
                    gold = batch["reply"]
                    k_label = batch["knowledge_grounding"]
                    p_label = batch["persona_grounding"]
                    persona_pred, k_index, r2_indices, r5_indices, pred= model.evaluate(batch)
                    
                    text_pred = pred
                    #text_pred = [pred]
                    text_target = self.tokenizer.decode(gold[0], skip_special_tokens=True)
              
                    
                    if text_target != "":
                        #debug
                        #print(persona_pred,p_label)
                        #persona_pred = 1 - p_label
                        self.k_accuracy.update(k_index[0], k_label)
                        self.p_accuracy.update(persona_pred, p_label)
                        self.recall_p.update(persona_pred, p_label)
                        self.precision_p.update(persona_pred, p_label)
                        self.f1_p.update(persona_pred, p_label)

                           
                        bleu += self.bleu_metric.compute(predictions=text_pred, references=[text_target])['score']
                        charf += self.chrf_metric.compute(predictions=text_pred, references=[text_target])['score']
                        r = self.rouge.score(text_pred[0], text_target)
                        #r = self.rouge.compute(predictions=[text_pred], references=[text_target])
                        rouge1 += r['rouge1'].fmeasure
                        rouge2 += r['rouge2'].fmeasure
                        rougel += r['rougeL'].fmeasure
                        #rouge1 += r['rouge1'].mid.fmeasure
                        #rouge2 += r['rouge2'].mid.fmeasure
                        #rougel += r['rougeL'].mid.fmeasure
                        unif1 += self.f1_uni.compute(predictions=text_pred, references=[text_target], word_order=1, char_order=0)['score']
                        qual_output = {
                            "dialogID": batch["dialogID"][0],
                            "landmark_link": batch["landmark_link"][0],
                            "dialog": batch["raw_dialog"][0],
                            "knowledge_pred": batch["raw_knowledge_cand"][0][k_index[0][0]],
                            "knowledge_pred_index": [k_index[0].detach().tolist()],
                            "knowledge_pred_index_r2": [r2_indices.detach().tolist()],
                            "knowledge_pred_index_r5": [r5_indices.detach().tolist()],
                            "knowledge_grounding": batch["raw_knowledge_cand"][0][k_label[0]],
                            "persona_pred": persona_pred.detach().tolist(),
                            "persona_grounding": p_label.detach().tolist(),
                            "persona_candidates": batch["raw_persona_cand"],
                            "predicted_utterance": text_pred,
                            "ground_truth_utterance": text_target
                        
                        }
                        qual_outputs.append(qual_output)
            
                json.dump({
                    "qualitative_results": qual_outputs
                }, fw, indent=2)
            
                # metric on all batches using custom accumulation
                knowledge_acc_f = self.k_accuracy.compute()
                persona_acc_f = self.p_accuracy.compute()
                precision_p = self.precision_p.compute()
                recall_p = self.recall_p.compute()
                pf1 = self.f1_p.compute()
                
                bleu4_f = bleu / len(qual_outputs)
                rouge1_f = rouge1 / len(qual_outputs)
                rouge2_f = rouge2 / len(qual_outputs)
                rougel_f = rougel / len(qual_outputs)
                charf_f = charf / len(qual_outputs)
                h2_f = h2 / len(qual_outputs)
                h5_f = h5 / len(qual_outputs)
                unif1_f = unif1 / len(qual_outputs)
            
                metrics = {
                    "k_acc": knowledge_acc_f.item(),
                    "p_acc": persona_acc_f.item(),
                    "p_f1": pf1.item(),
                    "hit@2": h2_f,
                    "hit@5": h5_f,
                    "bleu": bleu4_f,
                    "rouge1": rouge1_f,
                    "rouge2": rouge2_f,
                    "rougel": rougel_f,
                    "charf1": charf_f,
                    "unif1": unif1_f,
                    "p_precision": precision_p.item(),
                    "p_recall": recall_p.item(),
                }
            
                logging.info(
                    '%s Knowledge Accuracy: %2.5f | Persona Accuracy: %2.5f | Persona F1: %2.5f | P_Precision: %2.5f | P_Recall: %2.5f | BLEU : %2.5f | ROUGE1 : %2.5f | ROUGE2 : %2.5f | ROUGEL : %2.5f | CHARF1 : %2.5f | UniF1 : %2.5f'
                    % (typ, knowledge_acc_f.item(), persona_acc_f.item(), pf1, precision_p, recall_p, bleu4_f,
                       rouge1_f, rouge2_f, rougel_f, charf_f, unif1_f))
      
            return metrics

class Config:
    """コンフィグを表すクラス"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_model_and_tokenizer(config):
    """モデルとトークナイザーを読み込む関数"""
    # デバイスの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    print("Loading model and tokenizer...")
    print("Config:", config)

    # 辞書からオブジェクトに変換
    args = Config(**config)
    args.device = device

    model_class = MODEL_CLASSES[config["model"]]
    tokenizer = TOKENIZER_CLASSES[config["model"]].from_pretrained(config["backbone"])
    print("Tokenizer loaded successfully.")


    # トークン数の計算（retrieverとgeneratorに分ける場合）
    ret_orig_num_tokens = len(tokenizer.question_encoder)
    ret_num_added_tokens = 0  # 必要に応じて調整
    gen_orig_num_tokens = len(tokenizer.generator)
    gen_num_added_tokens = 0  # 必要に応じて調整

    add_tokens = {
        "retriever": ret_orig_num_tokens + ret_num_added_tokens,
        "generator": gen_orig_num_tokens + gen_num_added_tokens
    }

    # モデルの初期化
    model = model_class(args=args,tokenizer=tokenizer, num_tokens_to_add=add_tokens, device=device)
    print(args.backbone)
    # 学習済みモデルのパス
    model_path = config['load_pthpath']  # ここで .pth ファイルのパスを指定
    if model_path:
        print(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model' in checkpoint:
          state_dict = checkpoint['model']
        else:
          state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        print("Model weights loaded successfully.")


    #print("State dict keys (first 5):", list(model.state_dict().keys())[:5])
    # デバイスに移動
    model.to(device)
    
    return model, tokenizer,args

if __name__ == '__main__':
    # 設定ファイルの読み込み
    with open("/content/drive/MyDrive/experience/INFO/config/rag-tok-base-ct.json") as f:
        config = json.load(f)

    # モデルとトークナイザーを読み込む
    model, tokenizer,args = load_model_and_tokenizer(config)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # モデルとトークナイザーの初期化が成功したか確認
    print("Model and tokenizer loaded successfully.")
    task = config['task']  # 設定ファイルからタスクを取得
    model_type = config['model']  # 設定ファイルからモデルタイプを取得
    print(task,model_type)


    # DataProcessor や ModelProcessor の初期化
    #data_processor = DATA_PROCESSORS[config[args.task]](args, tokenizer)  # "pkchat" を想定
    #model_processor = MODEL_PROCESSORS[config[args.model]](args, tokenizer, args.device, data_processor, None)  # "rag-tok-ct" を想定
    data_processor = DATA_PROCESSORS[task](args, tokenizer)  # "pkchat" を想定
    model_processor = MODEL_PROCESSORS[model_type](args, tokenizer, args.device, data_processor, None)  # "rag-tok-ct" を想定

    # 評価オブジェクトを作成
    evaluation = Evaluation(args, tokenizer, data_processor, model_processor)

    # モデルを指定して評価を実行
    metrics = evaluation.evaluate(model, epoch=1, typ="test")  # validまたはtestを選択
    print(metrics)  # 結果を表示