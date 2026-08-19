import torch
from deepmd.pt.model.model.ener_model import EnergyModel
from deepmd.pt.model.descriptor.se_a import DescrptSeA
from deepmd.pt.model.task.ener import EnergyFittingNet

def main():
    print("=" * 60)
    print("测试 DeePMD 内置 EnergyModel（实例化对象）")
    print("=" * 60)

    type_map = ["O", "H"]

    descriptor = DescrptSeA(
        sel=[46, 92],
        rcut_smth=0.5,
        rcut=6.0,
        neuron=[25, 50, 100],
        axis_neuron=16,
        resnet_dt=False,
        seed=1,
    )

    fitting = EnergyFittingNet(
        dim_descrpt=descriptor.get_dim_out(),
        ntypes=len(type_map),
        n_hidden=[240, 240, 240],
        resnet_dt=True,
        seed=1,
    )

    print("正在实例化 EnergyModel...")
    model = EnergyModel(
        descriptor=descriptor,
        fitting=fitting,
        type_map=type_map,
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
    nloc = 6
    coord = torch.randn(nframes, nloc, 3, device=device)
    coord.requires_grad_(True)   # <--- 关键修复
    atype = torch.tensor([[0, 1, 1, 0, 1, 1]], device=device)
    box = torch.eye(3, device=device).unsqueeze(0) * 10.0

        # 注意：虽然我们用了 no_grad，但 coord 的梯度仍然会被追踪，
        # no_grad 只对前向操作有效，而 autograd.grad 仍然会运行。
        # 但为了安全，我们可以移除 no_grad 或保持，不影响。
    out = model(coord, atype, box=box)
    print("✅ 前向传播成功！")
    print(f"  输出 keys: {list(out.keys())}")
    if "energy" in out:
        print(f"  能量形状: {out['energy'].shape} -> {out['energy']}")
    if "force" in out:
        print(f"  力形状: {out['force'].shape}")

if __name__ == "__main__":
    main()