import torch
import yaml
from deepmd.pt.model.descriptor.se_a import DescrptSeA
from deepmd.pt.model.task.ener import EnergyFittingNet
from les import Les  # 请根据你的实际路径调整
from hybridles import HybridLESAtomicModel  # 导入你刚写的类
torch.set_default_tensor_type(torch.DoubleTensor)
from torch.utils.data import Dataset, DataLoader

from trydata import SyntheticWaterDataset
from train import Train

train_set = SyntheticWaterDataset(n_samples=80)
valid_set = SyntheticWaterDataset(n_samples=20, seed=123)  # 不同随机种子

# DataLoader
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=10, shuffle=False)


device = torch.device('cpu')
type_map = ["O", "H"]
rcut = 6.0
rcut_smth = 0.5
sel = [40, 40]

# 描述符（轻量化：输出 8x8 = 64 维）
descriptor = DescrptSeA(
    rcut=rcut,
    rcut_smth=rcut_smth,
    sel=sel,
    neuron=[8, 8],
    axis_neuron=8,
    precision="float64",
).to(device)

# 短程拟合网络（输入=64维，输出=1）
fitting_net = EnergyFittingNet(
    n_out=1,
    dim_descrpt=descriptor.get_dim_out(),
    n_hidden=[32, 16],
    ntypes=len(type_map),   # 这里是 2
    type_map=type_map,
    activation_function="tanh",
    precision="float64",
).to(device)

# LES 模型
with open('example/input.yaml', 'r') as f:
    les_config = yaml.safe_load(f)
les_config['use_atomwise'] = True  # 让 LES 内部通过 desc 生成 q
les_model = Les(les_arguments=les_config).to(device).float()


# ---------- 2. 构建混合模型 ----------
model = HybridLESAtomicModel(
    descriptor=descriptor,
    fitting_net=fitting_net,
    les_model=les_model,
    type_map=type_map,
).to(device)

'''
for batch in train_loader:
    coord = batch['coord'].to(device)
    atype = batch['atype'].to(device)
    cell = batch['cell'].to(device)
    # 前向传播
    E_pred = model(coord, atype, cell)
    print(f"预测能量形状: {E_pred.shape}, 数值: {E_pred}")
    break
'''

Train(model=model, train_loader=train_loader, device=device)