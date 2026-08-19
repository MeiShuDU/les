import torch
from deepmd.pt.model.descriptor.se_a import DescrptSeA
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

# 1. 设置设备
device = torch.device('cpu')

# 2. 设置参数
type_map = ["O", "H"]
rcut = 6.0
sel = [40, 40]

descriptor = DescrptSeA(
    rcut=rcut,
    rcut_smth=0.5,
    sel=sel,
    axis_neuron=8,
).to(device)


# 4. 准备输入数据
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

# 5. 添加批次维度
coord = coord.unsqueeze(0)  # [1, 6, 3]
atype = atype.unsqueeze(0)  # [1, 6]
cell = cell.unsqueeze(0)    # [1, 3, 3]

# 6. 构建邻居列表
extended_coord, extended_atype, mapping, nlist = extend_input_and_build_neighbor_list(
    coord, atype, rcut, sel, box=cell
)

# 7. 计算描述符
desc = descriptor(extended_coord, extended_atype, nlist)

# 8. 输出结果
print(f"描述符向量维度: {descriptor.get_dim_out()}")
print(f"输出形状: {desc[0].shape}")
#print(desc)