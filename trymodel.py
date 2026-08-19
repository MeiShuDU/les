import torch
import sys
from pathlib import Path
torch.set_default_tensor_type(torch.DoubleTensor)

# 确保可以导入 les_plugin
#sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepmd.pt.model.descriptor.se_a import DescrptSeA
from deepmd.pt.model.task.ener import EnergyFittingNet
from hybridles_model import HybridLESModel

def main():
    print("=" * 60)
    print("测试 HybridLESModel")
    print("=" * 60)

    type_map = ["O", "H"]

    # 手动实例化描述符
    descriptor = DescrptSeA(
        sel=[46, 92],
        rcut_smth=0.5,
        rcut=6.0,
        neuron=[25, 50, 100],
        axis_neuron=16,
        resnet_dt=False,
        seed=1,
    )

    # 手动实例化拟合网络
    fitting = EnergyFittingNet(
        dim_descrpt=descriptor.get_dim_out(),
        ntypes=len(type_map),
        n_hidden=[240, 240, 240],
        resnet_dt=True,
        seed=1,
    )

    les_params = {
        "use_atomwise": True,
        "sigma": 1.0,
        "dl": 1.5,
    }

    print("正在实例化 HybridLESModel...")
    model = HybridLESModel(
        descriptor=descriptor,
        fitting=fitting,
        type_map=type_map,
        les_params=les_params,
    )
    print("✅ 模型实例化成功！")

    print("\n模型信息:")
    print(f"  - 类型映射: {model.get_type_map()}")
    print(f"  - 截断半径: {model.get_rcut()}")
    print(f"  - 选择原子数: {model.get_sel()}")
    #print(f"  - 输出维度: {model.get_dim_out()}")

    print("\n测试前向传播（随机数据）...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    nframes = 1
    nloc = 6  # 两个水分子
    coord = torch.randn(nframes, nloc, 3, device=device)
    coord.requires_grad_(True)   # 必须开启，否则自动微分失败
    atype = torch.tensor([[0, 1, 1, 0, 1, 1]], device=device)
    box = torch.eye(3, device=device).unsqueeze(0) * 10.0

    out = model(coord, atype, box=box)
    print("✅ 前向传播成功！")
    print(f"  输出 keys: {list(out.keys())}")
    if "energy" in out:
        print(f"  能量形状: {out['energy'].shape} -> {out['energy']}")
    if "force" in out:
        print(f"  力形状: {out['force'].shape}")

if __name__ == "__main__":
    main()