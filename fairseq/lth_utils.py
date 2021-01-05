import logging, os
from fairseq import checkpoint_utils
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.prune as prune
from typing import Any, Dict
import pdb


def get_weights_to_prune(state_dict):
    weights=[]
    for key in state_dict.keys():
        if 'weight' in key or 'weight_orig' in key:
            if not "layer_norm" in key and "layer" in key:
                weights.append(key)
    return weights


def get_weights_to_prune_tuple(m, prefix, weights):
    if not prefix =="":
        prefix = prefix +"."
    if hasattr(m, "weight") and not "layer_norm" in prefix and "layer" in prefix:
        print("ADD", prefix)
        weights.append((m, "weight"))
    else:
        for n, c in m.named_children():
            print(prefix, n)
            weights += get_weights_to_prune_tuple(c, prefix+n, weights)

    return weights

def get_global_weights_to_prune(m):
    weights = m.get_global_weights_to_prune()   
    return weights
    #return get_weights_to_prune_tuple(m, "", weights)



def upgrade_state_dict_with_checkpoint_weights(
    state_dict: Dict[str, Any], weights, checkpoint: str
) -> Dict[str, Any]:
    if not os.path.exists(checkpoint):
        raise IOError("Model file not found: {}".format(checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint)
   
    ch_state_dict = state["model"]
    for ckey in ch_state_dict.keys():
        for key in [ckey, ckey.replace("decoder.","")]:
            if weights==None or key in weights :
                if key in state_dict:
                    state_dict[key] = ch_state_dict[ckey]
                elif "{}_orig".format(key) in state_dict:
                    state_dict["{}_orig".format(key)] = ch_state_dict[ckey]
                else:
                    print("WARNING: unknown key {}".format(key))
    #pdb.set_trace()
    return state_dict


def do_global_prune(m, pruning, weights=None):
    if not weights:
        weights = tuple(get_global_weights_to_prune(m))
    else:
        weights = tuple(weights)

    prune.global_unstructured(
        weights,
        pruning_method=prune.L1Unstructured,
        amount=pruning
    )

def do_prune(m, prefix, pruning):
    if not prefix =="":
        prefix = prefix +"."
    for n, c in m.named_children():
        #print(prefix, n)
        do_prune(c, prefix+n, pruning)
    if hasattr(m, "weight") and not "layer_norm" in prefix:
        #print("PRUNE weight of ", prefix)
        #parameters_to_prune.append((m, "weight"))
        prune.l1_unstructured(m, name="weight", amount=pruning)

def get_pruning_per_weight(m, prefix):
    if not prefix =="":
        prefix = prefix +"."
    for n, c in m.named_children():
        #print(prefix, n)
        get_pruning_per_weight(c, prefix+n)

    if hasattr(m, "weight"):
        #        print(prefix)
        print("Sparsity in {}.weight: {:.2f}%".format(prefix,
            100. * float(torch.sum(m.weight == 0))/ float(m.weight.nelement()) ))


def get_lottery_ticket(model, init_checkpoint, final_checkpoint, pruning, it=0):
    weights = get_weights_to_prune(model.state_dict())
    # get weight values from final checkpoint
    if it==0:
        final_loaded_state_dict = upgrade_state_dict_with_checkpoint_weights(
            state_dict=model.state_dict(),
            weights=weights,
            checkpoint = final_checkpoint
        )
        
        model.load_state_dict(final_loaded_state_dict, strict=True)


    #model.do_prune(pruning)

    do_global_prune(model, pruning)


    #prune.global_unstructured(parameters_to_prune, 
    #                          pruning_method=prune.L1Unstructured,
    #                          amount=self.pruning)
#    get_pruning_per_weight(model, "")
    # get weight values from initialization checkpoint
        
    initial_state_dict = upgrade_state_dict_with_checkpoint_weights(
        state_dict=model.state_dict(),
        weights=weights,
        checkpoint=init_checkpoint,
    )
            
    model.load_state_dict(initial_state_dict, strict=True)
    return model

# make pruning final
def finalize_prune(m, prefix):
    if not prefix =="":
        prefix = prefix +"."
    for n, c in m.named_children():
        #print(prefix, n)
        finalize_prune(c, prefix+n)
    if hasattr(m, "weight_orig"):
        #        print(prefix)
        prune.remove(m, "weight")
            #finalize_prune(self, "")
