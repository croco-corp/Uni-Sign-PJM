import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset, S2T_Dataset_PJM
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from models import get_requires_grad_dict
from SLRT_metrics import translation_performance, islr_performance, wer_list
from transformers import get_scheduler
from config import *

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)

    if args.wandb and WANDB_AVAILABLE and utils.is_main_process():
        os.makedirs(args.wandb_dir, exist_ok=True)
        wandb.init(
            project=args.wandb_project,
            name=os.path.basename(args.output_dir),
            config=vars(args),
            dir=args.wandb_dir,
        )

    print(f"Creating dataset:")
    if args.dataset == "PJM":
        suffix = {'si': '', 'ms': '_ms', 'filtered': '_filtered'}[args.pjm_split]
        print(f"  PJM split family: {args.pjm_split} (suffix='{suffix}')")
        train_data = S2T_Dataset_PJM(
            path=f'../CrocoSign/data/split_train{suffix}.csv',
            texts_path='../CrocoSign/data/texts_eng.h5',
            args=args,
            phase='train'
        )
        dev_data = S2T_Dataset_PJM(
            path=f'../CrocoSign/data/split_val{suffix}.csv',
            texts_path='../CrocoSign/data/texts_eng.h5',
            args=args,
            phase='val'
        )
        test_data =  S2T_Dataset_PJM(
            path=f'../CrocoSign/data/split_test{suffix}.csv',
            texts_path='../CrocoSign/data/texts_eng.h5',
            args=args,
            phase='test'
        )

    else: 
        train_data = S2T_Dataset(path=train_label_paths[args.dataset], 
                                args=args, phase='train')
        test_data = S2T_Dataset(path=test_label_paths[args.dataset], 
                        args=args, phase='test')
    print(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem,
                                 drop_last=True)

    print(test_data)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)

    if "How2Sign" not in args.dataset and args.dataset != "PJM":
        dev_data = S2T_Dataset(path=dev_label_paths[args.dataset],
                               args=args, phase='dev')
        print(dev_data)
        # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
        dev_sampler = torch.utils.data.SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    collate_fn=dev_data.collate_fn,
                                    sampler=dev_sampler,
                                    pin_memory=args.pin_mem)
    elif args.dataset == "PJM":
        print(dev_data)
        dev_sampler = torch.utils.data.SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    collate_fn=dev_data.collate_fn,
                                    sampler=dev_sampler,
                                    pin_memory=args.pin_mem)
    else:
        dev_dataloader = test_dataloader

    print(f"Creating model:")
    model = Uni_Sign(args=args)

    model.cuda()
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']

        ret = model.load_state_dict(state_dict, strict=False) # in phase 3 comeback to True
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    
    model_without_ddp = model
    if args.freeze_visual:
        visual_attrs = [
            "proj_linear", "gcn_modules", "fusion_gcn_modules",
            "part_para", "pose_proj",
            "rgb_support_backbone", "rgb_proj",
            "fusion_pose_rgb_linear", "fusion_pose_rgb_DA", "fusion_gate",
        ]
        frozen_count = 0
        for attr in visual_attrs:
            mod = getattr(model, attr, None)
            if mod is None:
                continue
            if isinstance(mod, torch.nn.Parameter):
                mod.requires_grad = False
                frozen_count += 1
            else:
                for p in mod.parameters():  
                    p.requires_grad = False
                    frozen_count += 1
        print(f"[phase2] visual path frozen ({frozen_count} param tensors)")

    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType
        targets = [t.strip() for t in args.lora_target.split(",") if t.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=targets,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model.mt5_model = get_peft_model(model.mt5_model, lora_cfg)
        model.mt5_model.print_trainable_parameters()
        print(f"[phase2] mT5 wrapped with LoRA targets={targets}")

        if args.lora_ckpt:
            lora_state = torch.load(args.lora_ckpt, map_location='cpu')['model']
            ret = model.load_state_dict(lora_state, strict=False)
            print(f"[phase2] loaded LoRA ckpt: missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}")

        
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )
    
    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    model_without_ddp = model.module.module
    # print(model_without_ddp)
    print(optimizer)

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0
    if args.task == "CSLR":
        max_accuracy = 1000
    
    if args.eval:
        if utils.is_main_process():
            if args.task != "ISLR" and "How2Sign" not in args.dataset:
                print("📄 dev result")
                dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
                if args.wandb and WANDB_AVAILABLE:
                    wandb.log({f'dev_{k}': v for k, v in dev_stats.items()})
            print("📄 test result")
            test_stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test')
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({f'test_{k}': v for k, v in test_stats.items()})

        if args.wandb and WANDB_AVAILABLE and utils.is_main_process():
            wandb.finish()
        return
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': get_requires_grad_dict(model_without_ddp),
                }, checkpoint_path)

        # single gpu inference
        if utils.is_main_process():
            test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
            if epoch == args.epochs - 1:
                final_test_stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test')
                if args.wandb and WANDB_AVAILABLE:
                    wandb.log({f'final_test_{k}': v for k, v in final_test_stats.items()})
                if args.output_dir:
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps({f'final_test_{k}': v for k, v in final_test_stats.items()}) + "\n")

            if args.task == "SLT":
                if max_accuracy < test_stats["bleu4"]:
                    max_accuracy = test_stats["bleu4"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}")
                print(f'Max BLEU-4: {max_accuracy:.2f}%')
            
            elif args.task == "ISLR":
                if max_accuracy < test_stats["top1_acc_pi"]:
                    max_accuracy = test_stats["top1_acc_pi"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(f"PI accuracy of the network on the {len(dev_dataloader)} dev videos: {test_stats['top1_acc_pi']:.2f}")
                print(f'Max PI accuracy: {max_accuracy:.2f}%')
            
            elif args.task == "CSLR":
                if max_accuracy > test_stats["wer"]:
                    max_accuracy = test_stats["wer"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)
                            
                print(f"WER of the network on the {len(dev_dataloader)} dev videos: {test_stats['wer']:.2f}")
                print(f'Min WER: {max_accuracy:.2f}%')
        
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if args.wandb and WANDB_AVAILABLE:
                wandb.log(log_stats)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.wandb and WANDB_AVAILABLE and utils.is_main_process():
        wandb.finish()

def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if target_dtype != None:
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(target_dtype).cuda()

        if args.task == "CSLR":
            tgt_input['gt_sentence'] = tgt_input['gt_gloss']
        stack_out = model(src_input, tgt_input)
        
        total_loss = stack_out['loss']
        model.backward(total_loss)
        model.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if args.wandb and WANDB_AVAILABLE and utils.is_main_process():
            wandb.log({"train_loss_step": loss_value, "lr_step": optimizer.param_groups[0]["lr"]})

        if args.quick_break > 0 and (step + 1) >= args.quick_break:
            print(f"[quick_break] stopping epoch after {step+1} steps (quick_break={args.quick_break})")
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16
        
    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
        tgt_name = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype != None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()
            
            if args.task == "CSLR":
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']
            stack_out = model(src_input, tgt_input)
            
            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())
        
            output = model_without_ddp.generate(stack_out, 
                                                max_new_tokens=100, 
                                                num_beams = 4,
                        )

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])
                tgt_name.append(src_input['name_batch'][i])

    tokenizer = model_without_ddp.mt5_tokenizer
    padding_value = tokenizer.eos_token_id
    
    pad_tensor = torch.ones(150-len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0],pad_tensor.long()),dim = 0)

    tgt_pres = pad_sequence(tgt_pres,batch_first=True,padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    # fix mt5 tokenizer bug
    if args.dataset == 'CSL_Daily' and args.task == "SLT":
        tgt_pres = [' '.join(list(r.replace(" ",'').replace("\n",''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

    if args.task == "SLT":
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
        for k,v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters['rouge'].update(rouge_score)

        n_ex = min(args.num_examples, len(tgt_pres))
        if n_ex > 0:
            print(f'--- {n_ex} example translations ({phase}) ---')
            for i in range(n_ex):
                name_i = tgt_name[i] if i < len(tgt_name) else f'idx{i}'
                print(f'[{name_i}]')
                print(f'  ref: {tgt_refs[i]}')
                print(f'  hyp: {tgt_pres[i]}')
            print('--- end examples ---')

        if args.bertscore:
            try:
                from bert_score import score as bert_score_fn
                P, R, F1 = bert_score_fn(tgt_pres, tgt_refs, lang='en', rescale_with_baseline=False, verbose=False)
                bertscore_f1 = float(F1.mean().item()) * 100
                print(f'BERTScore F1: {bertscore_f1:.2f}')
                metric_logger.meters['bertscore_f1'].update(bertscore_f1)
            except ImportError:
                print('[warn] --bertscore set but `bert_score` package is not installed; skipping')

        if args.eval and (args.dataset == 'How2Sign' or args.dataset == 'OpenASL'):
            # BLEURT # follow GloFE
            # Due to the long processing time, only --eval will be executed.
            from bleurt import score
            checkpoint = "./BLEURT-20"
            scorer = score.BleurtScorer(checkpoint)
            scores_bleurt = scorer.score(references=tgt_refs[:], candidates=tgt_pres[:])
            print('BLEURT:', sum(scores_bleurt)/len(scores_bleurt))

    elif args.task == "ISLR":
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
        metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)
        
    elif args.task == "CSLR":
        wer_results = wer_list(hypotheses=tgt_pres, references=tgt_refs)
        print(wer_results)
        for k,v in wer_results.items():
            metric_logger.meters[k].update(v)

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        pres_path = args.output_dir + f'/{phase}_tmp_pres.txt'
        refs_path = args.output_dir + f'/{phase}_tmp_refs.txt'
        with open(pres_path, 'w') as f:
            for i in range(len(tgt_pres)):
                f.write(f"sample: {tgt_name[i]}, prediction: " + tgt_pres[i]+'\n')
        with open(refs_path, 'w') as f:
            for i in range(len(tgt_refs)):
                f.write(f"sample: {tgt_name[i]}, ground-truth: " + tgt_refs[i]+'\n')

        if args.wandb and WANDB_AVAILABLE and args.task == "SLT":
            n_ex = min(args.num_examples, len(tgt_pres))
            ex_table = wandb.Table(columns=["name", "ref", "hyp"])
            for i in range(n_ex):
                ex_table.add_data(str(tgt_name[i]), tgt_refs[i], tgt_pres[i])
            full_table = wandb.Table(columns=["name", "ref", "hyp"])
            for i in range(len(tgt_pres)):
                full_table.add_data(str(tgt_name[i]), tgt_refs[i], tgt_pres[i])
            wandb.log({
                f'{phase}_examples': ex_table,
                f'{phase}_all_predictions': full_table,
            })
            wandb.save(pres_path, base_path=args.output_dir, policy='now')
            wandb.save(refs_path, base_path=args.output_dir, policy='now')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)