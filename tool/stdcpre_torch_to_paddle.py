import torch
import paddle
import paddle.fluid as fluid
from collections import OrderedDict
from nets.stdcnet import STDCNet813,STDCNet1446
from nets.stdcnet_paddle import STDCNet1446 as STDCNet1446_paddle
from nets.stdcnet_paddle import STDCNet813 as STDCNet813_paddle
from models.model_stages_paddle import BiSeNet

def convert_params(params_path):
    import torch
    import paddle
    params = torch.load(params_path, map_location=torch.device('cpu'))
    new_params = dict()
    bn_w_name_list = list()
    for k, v in params.items():
        print(k,v)
        if k.endswith(".running_mean"):
            new_params[k.replace(".running_mean", "._mean")] = v.detach().numpy(
            )
        elif k.endswith(".running_var"):
            new_params[k.replace(".running_var", "._variance")] = v.detach(
            ).numpy()
            bn_w_name_list.append(k.replace(".running_var", ".weight"))
        else:
            new_params[k] = v.detach().numpy()
    for k, v in new_params.items():
        if len(v.shape) == 2 and k.endswith(
                ".weight") and k not in bn_w_name_list:
            new_params[k] = v.T
    paddle.save(new_params,
                params_path.replace(".pth", ".pdiparams").replace(
                    ".pt", ".pdiparams").replace(".ckpt", ".pdiparams"))



if __name__=='__main__':
    pth = torch.load("STDCNet813M_73.91.tar",map_location=torch.device('cpu'))
    torch.save(pth["state_dict"], "STDCNet813M_73.91.pth")
    convert_params('STDCNet813M_73.91.pth')
