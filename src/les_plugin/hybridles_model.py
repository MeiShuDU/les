# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Any, Optional, Dict
import torch
from deepmd.pt.model.model.model import BaseModel
from deepmd.pt.model.model.make_model import make_model
from deepmd.pt.model.model.dp_model import DPModelCommon
from les import Les
from hybridles import HybridLESAtomicModel
#from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

HybridLESModel_ = make_model(HybridLESAtomicModel)

@BaseModel.register("hybrid_ener")
class HybridLESModel(DPModelCommon, HybridLESModel_):
    model_type = "hybrid_ener"

    def __init__(
        self,
        descriptor,
        fitting,
        type_map: list[str],
        les_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        atomic_model = HybridLESAtomicModel(
            descriptor=descriptor,
            fitting=fitting,
            type_map=type_map,
            les_params=les_params,
        )
        HybridLESModel_.__init__(self, atomic_model_=atomic_model, **kwargs)
        DPModelCommon.__init__(self)
        # 保存描述符实例以供 forward 重新计算描述符
        self.descriptor = descriptor

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        # 1. 调用父类 forward_common 获得短程结果（能量、力等）
        from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )

        # 2. 重新计算描述符（用于 LES）
        rcut = self.get_rcut()
        sel = self.get_sel()
        extended_coord, extended_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            coord, atype, rcut, sel, box=box
        )
        desc = self.descriptor(extended_coord, extended_atype, nlist)[0]  # [nframes, nloc, dim]

        # 3. 计算 LES 长程能量和力
        nframes = coord.shape[0]
        nloc = coord.shape[1]
        device = coord.device
        coord.requires_grad_(True)   # 确保坐标有梯度

        E_lr_list = []
        force_lr_list = []
        for i in range(nframes):
            coord_i = coord[i]   # [nloc, 3]
            desc_i = desc[i]     # [nloc, dim]
            cell_i = box[i].reshape(3, 3).unsqueeze(0) if box is not None else None

            # 调用 LES 模型（已保存在 atomic_model 中）
            les_out = self.atomic_model.les_model(
                positions=coord_i,
                cell=cell_i,
                desc=desc_i,
                batch=None,
                compute_energy=True,
            )
            E_lr = les_out['E_lr']  # 标量
            # 计算力
            force_lr = -torch.autograd.grad(
                E_lr, coord_i,
                create_graph=True,
                retain_graph=True,
            )[0]  # [nloc, 3]

            E_lr_list.append(E_lr)
            force_lr_list.append(force_lr)

        E_lr_total = torch.stack(E_lr_list)  # [nframes]
        force_lr_total = torch.stack(force_lr_list)  # [nframes, nloc, 3]

        # 4. 合并结果
        model_predict = {}
        model_predict["energy"] = model_ret["energy_redu"] + E_lr_total
        model_predict["atom_energy"] = model_ret["energy"]  # 短程原子能量（不含长程）
        if self.do_grad_r("energy"):
            force_sr = model_ret["energy_derv_r"].squeeze(-2)  # [nframes, nloc, 3]
            model_predict["force"] = force_sr + force_lr_total
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict
