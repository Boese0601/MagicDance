from typing import Union, Dict, List, Tuple, Any, Callable
import glob
import os
import io
import re
import torch
import shutil
import time
from dataset.hdfs_io import hexists, hmkdir, hopen, hcopy

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        print("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def ModelSaver(output_dir, model, optimizer=None):
    output_model_file = os.path.join(output_dir, "model.pt")
    state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.state_dict().items()}
    torch.save(state_dict, output_model_file)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), f'{output_dir}/optimizer.pt')


# checkpointer from fex. Save to hdfs
def save(obj, filepath: str, **kwargs):
    """ save model """
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "wb") as writer:
            torch.save(obj, writer, **kwargs)
    else:
        torch.save(obj, filepath, **kwargs)


def save_checkpoint(output_dir: str,
                    epoch: Union[int, str],
                    lg_loss_scale: float,
                    model_state: Dict[str, Any],
                    training_states: Dict[str, Any],
                    num_serialized_models_to_keep: int = 20) -> None:
    """
    保存 checkpoint到remote hdfs中：
    args:
        epoch: 当前训练的epoch数
        model_state: 当前训练model的参数
        training_states: 当前训练的参数
        is_best_so_far: 当前是否save的checkpoint是否为最优
    """
    if not hexists(output_dir):
        hmkdir(output_dir)
    
    model_path = os.path.join(output_dir, "model_state-{}.th".format(epoch))
    training_path = os.path.join(output_dir, "optimizer_state_latest.th")
    save(model_state["cur_model"], model_path)
    save({"state_dict": training_states, "step": epoch, "lg_loss_scale": lg_loss_scale}, training_path)



def save_checkpoint_ema(output_dir: str,
                    epoch: Union[int, str],
                    models_state: Dict[str, Any],
                    optimizer_state: Dict[str, Any],
                    scaler_state: Dict[str, Any],
                    num_serialized_models_to_keep: int = 20) -> None:
    """
    保存 checkpoint到remote hdfs中：
    args:
        epoch: 当前训练的epoch数
        model_state: ema 参数, 0: 当前训练model的参数, others: ema_model 参数
        training_states: 当前训练的参数
        is_best_so_far: 当前是否save的checkpoint是否为最优
    """
    if not hexists(output_dir):
        hmkdir(output_dir)
    for ema_rate, model_state in models_state.items():
        if ema_rate == 0:
            # 0: 当前训练model的参数
            model_path = os.path.join(output_dir, "model_state-{}.th".format(epoch))
        elif isinstance(ema_rate, str):
            # some other keyword
            model_path = os.path.join(output_dir, "model_state-{}_{}.th".format(epoch, ema_rate))
        else:
            model_path = os.path.join(output_dir, "model_state-{}_ema_{}.th".format(epoch, ema_rate))
        save(model_state, model_path)

    
    training_path = os.path.join(output_dir, "optimizer_state_latest.th")
    if hexists(training_path):
        hcopy(training_path, os.path.join(output_dir, f"optimizer_state_{epoch}.th"))

    save({"state_dict": optimizer_state, "step": epoch, "scaler_state_dict": scaler_state}, training_path)


# resuming training
def load(filepath: str, **kwargs):
    """ load model """
    if not filepath.startswith("hdfs://"):
        return torch.load(filepath, **kwargs)
    with hopen(filepath, "rb") as reader:
        accessor = io.BytesIO(reader.read())
        state_dict = torch.load(accessor, **kwargs)
        del accessor
        return state_dict

def load_from_pretrain(pretrain_path, map_location='cpu'):

    # partial load pretrain state dict
    pretrain_state_dict_parsed = {}
    if pretrain_path != "":
        pretrain_state_dict = load(pretrain_path, map_location=map_location)
    return pretrain_state_dict