import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from deepmd.pt.model.atomic_model import BaseAtomicModel
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

from deepmd.dpmodel import FittingOutputDef
from deepmd.dpmodel.output_def import OutputVariableCategory, OutputVariableOperation

class HybridLESAtomicModel(BaseAtomicModel, nn.Module):
    def __init__(
        self,
        descriptor: nn.Module,
        fitting_net: nn.Module,
        les_model: nn.Module,
        type_map: list[str],
    ):
        super().__init__(type_map=type_map)
        self.descriptor = descriptor
        self.fitting_net = fitting_net
        self.les_model = les_model
        self.type_map = type_map
        self.ntypes = len(type_map)

        self.rcut = descriptor.get_rcut()
        self.sel = descriptor.get_sel()
        self.sel_type = list(range(self.ntypes))

        self.bias_keys = ["energy"]   # 指定要统计的输出键
        self.out_bias = None          # 将在后续统计中设置
        self.out_std = None

    # ---------- 抽象方法实现 ----------
    def get_rcut(self) -> float:
        return self.rcut

    def get_sel(self) -> List[int]:
        return self.sel

    def get_type_map(self) -> List[str]:
        return self.type_map

    def get_dim_out(self) -> int:
        return self.fitting_net.get_dim_out()

    def get_dim_descrpt(self) -> int:
        return self.descriptor.get_dim_out()

    def get_dim_aparam(self) -> int:
        return 0

    def get_dim_fparam(self) -> int:
        return 0

    def get_sel_type(self) -> List[int]:
        return self.sel_type

    def has_message_passing(self) -> bool:
        return False

    def is_aparam_nall(self) -> bool:
        return False

    def mixed_types(self) -> bool:
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        return False

    def set_case_embd(self, case_embd: Optional[torch.Tensor] = None):
        pass

    def fitting_output_def(self):
        return [
            {
                "key": "energy",
                "size": 1,
                "category": "energy",
                "reduce": "sum",
            }
        ]
    def atomic_output_def(self) -> FittingOutputDef:
        from deepmd.dpmodel.output_def import OutputVariableCategory, OutputVariableOperation
        return FittingOutputDef(
            {
                "energy": {
                    "shape": (1,),
                    "category": OutputVariableCategory.ATOMIC,
                    "operation": OutputVariableOperation.NONE,
                },
                "mask": {
                    "shape": (1,),
                    "category": OutputVariableCategory.ATOMIC,
                    "operation": OutputVariableOperation.NONE,
                },
        }
    )

    def apply_out_stat(self, ret_dict, atype):
        if self.out_bias is None or self.out_std is None:
            return ret_dict
        return super().apply_out_stat(ret_dict, atype)

    def forward_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
        
    ) -> Dict[str, torch.Tensor]:
        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc]
        print(f'shape of nlist : {nlist.shape}')
        print(f'shape of ex_coord : {extended_coord.shape}')
        print(f"nloc = {nloc}")
        print(f"atype = {atype}")
        self.desc = self.descriptor(extended_coord, extended_atype, nlist)[0]
        E_sr_atom = self.fitting_net(self.desc, atype)['energy']
        mask = torch.ones(
            self.desc.shape[0], nloc, dtype=torch.bool, device=extended_coord.device
    )

        return {'energy':E_sr_atom, 'mask' : mask}

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        cell: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        coord.requires_grad_(True)
        batch_size = coord.shape[0]

        extended_coord, extended_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            coord, atype, self.rcut, self.sel, box=cell
        )

        atomic_out = self.forward_atomic(extended_coord, extended_atype, atype, nlist, mapping)
        E_sr_atom = atomic_out['energy']
        E_sr = E_sr_atom.sum(dim=1)

        E_lr_list = []
        for i in range(batch_size):
            coord_i = coord[i]
            cell_i = cell[i].unsqueeze(0)
            desc_i = self.desc[i]

            les_out = self.les_model(
                positions=coord_i,
                cell=cell_i,
                desc=desc_i,
                batch=None,
                compute_energy=True,
            )
            E_lr_list.append(les_out['E_lr'])
        E_lr = torch.stack(E_lr_list)
        E_tot = E_sr + E_lr
        force_pred = -torch.autograd.grad(
            E_tot.sum(),
            coord,
            create_graph=True,
            retain_graph=True
        )[0]

        return {'energy' : E_tot, 'force' : force_pred}