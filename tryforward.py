import torch
import yaml
from deepmd.pt.model.descriptor.se_a import DescrptSeA
from deepmd.pt.model.task.ener import EnergyFittingNet
from les import Les  # 请根据你的实际路径调整
from hybridles11 import HybridLESAtomicModel  # 导入你刚写的类
torch.set_default_tensor_type(torch.DoubleTensor)

# ---------- 1. 实例化组件 ----------
device = torch.device('cuda')
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

# ---------- 3. 准备输入数据（两个水分子） ----------
coord = torch.tensor([
    [0.0, 0.0, 0.0],
    [0.76, 0.59, 0.0],
    [-0.76, 0.59, 0.0],
    [3.0, 0.0, 0.0],
    [3.76, 0.59, 0.0],
    [2.24, 0.59, 0.0],
], dtype=torch.float64, device=device)

atype = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.int32, device=device)
cell = torch.diag(torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64, device=device))

# 添加 batch 维度
coord = coord.unsqueeze(0)   # [1, 6, 3]
atype = atype.unsqueeze(0)   # [1, 6]
cell = cell.unsqueeze(0)     # [1, 3, 3]

# ---------- 4. 预测总能量 ----------

E_pred = model(coord, atype, cell)
print(E_pred)