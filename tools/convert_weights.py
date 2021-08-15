import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import paddle


def torch_dict_to_paddle(torch_dict):
    paddle_dict = {}
    suffix_map = {".running_mean": "._mean", ".running_var": "._variance", ".num_batches_tracks": None}
    for key in torch_dict:
        paddle_key = key
        weight = torch_dict[key].detach().cpu().numpy()
        if key.startswith('fc'):
            weight = weight.transpose()
        else:
            for suffix_key in suffix_map.keys():
                if key.endswith(suffix_key):
                    if suffix_map[suffix_key]:
                        paddle_key = key[:-len(suffix_key)] + suffix_map[suffix_key]
                    else:
                        paddle_key = None
        if paddle_key:
            paddle_dict[paddle_key] = weight
    return paddle_dict


torch_state_dict = torch.load("model/yolox_darknet53.pth")['model']
print(torch_state_dict.keys())

paddle_state_dict = torch_dict_to_paddle(torch_state_dict)
paddle.save(paddle_state_dict, "model/yolox_darknet53.pdparams")
