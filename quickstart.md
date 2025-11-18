# PD-CLM Quickstart

## Kurulum
- `pip install -r requirements.txt`
- İsteğe bağlı izleme: W&B hesabında oturum aç (`wandb login`)

## Veri ve KB İndeksleme
- Türkçe Wikipedia (Parquet):
```
python download_data.py --dataset wikipedia --split train --subset_pct 5 --build_index true --chunk_size 2000 --embed_dim 512 --wiki_date 20231101.tr
```
- İngilizce Wikipedia (isteğe bağlı):
```
python download_data.py --dataset wikipedia --split train --subset_pct 5 --build_index true --chunk_size 2000 --embed_dim 512 --wiki_date 20231101.en
```
- Çıktılar:
  - Metin: `data/raw/wikipedia_tr_sample.txt`
  - İndeks: `data/index/faiss.index`, `chunks.jsonl`, `meta.json`

## Faz-1: Pre-training (T4 için güvenli)
- TF32 hız/kararlılık (Ampere):
```
python -c "import torch; torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; import subprocess,sys; sys.exit(subprocess.call(['python','faz1_training_fixed.py','--data-path','data/raw/wikipedia_tr_sample.txt','--iterations','20000','--batch-size','2500','--accumulation-steps','16','--val-chars','10000','--embed-dim','128','--num-layers','3','--heads','2','--window-size','256','--max-windows','16','--wandb','--save-every','1000','--checkpoint-dir','checkpoints/faz1_tr'])) )"
```
- OOM olursa kademeli azalt: `--batch-size 3000→2000`, `--window-size 256→224`, `--embed-dim 128→96`, `--max-windows 16→12`

## Faz-2: SFT (Araçlı)
```
python -c "from src.pdclm import PDCLM, instruction_sft_train; from src.utils import load_checkpoint; import torch; m=PDCLM(); opt=torch.optim.AdamW(m.parameters()); load_checkpoint(m,opt,'checkpoints/faz1_tr/faz1_last.pt'); instruction_sft_train(m, num_samples=10000, epochs=2, use_tools=True)"
```

## Faz-3: PPO/RL
```
python -c "from src.pdclm import PDCLM, ppo_train; from src.utils import load_checkpoint; import torch; m=PDCLM(); opt=torch.optim.AdamW(m.parameters()); load_checkpoint(m,opt,'checkpoints/pdclm_sft_final.pt'); ppo_train(m, epochs=2, tasks_per_epoch=500, use_tools=True, gen_max_bytes=128, cot_max_steps=5)"
```

## Değerlendirme (GSM8K)
```
python -c "from src.pdclm import PDCLM; from src.utils import evaluate_gsm8k, load_checkpoint; import torch; m=PDCLM(); opt=torch.optim.AdamW(m.parameters()); load_checkpoint(m,opt,'checkpoints/pdclm_ppo_final.pt'); print(evaluate_gsm8k(m, split='test', sample_size=200))"
```

## DDP (Çoklu GPU)
- Örnek (2x T4):
```
torchrun --nproc_per_node=2 faz1_training_fixed.py --data-path data/raw/wikipedia_tr_sample.txt --iterations 20000 --batch-size 2500 --accumulation-steps 16 --val-chars 10000 --embed-dim 128 --num-layers 3 --heads 2 --window-size 256 --max-windows 16 --wandb --save-every 1000 --checkpoint-dir checkpoints/faz1_tr
```

## Sorun Giderme
- OOM: `--max-windows`, `--window-size`, `--embed-dim`, `--batch-size` düşür; kümülatif `--accumulation-steps` artır.
- FAISS: `download_data.py` komutu sonunda `[INDEX] Saved FAISS index ...` çıktısını doğrula.
- W&B erişimi: Proje görünürlüğünü aç veya oturum kontrol et.