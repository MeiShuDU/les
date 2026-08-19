import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from deepmd.pt.model.atomic_model import BaseAtomicModel
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

class HybridLESAtomicModel(BaseAtomicModel, nn.Module):
#class HybridLESAtomicModel(nn.Module):
    def __init__(
        self,
        descriptor: nn.Module,
        fitting_net: nn.Module,
        les_model: nn.Module,
        type_map: list[str],
    ):
        super().__init__(type_map=type_map)
        #super().__init__()
        self.descriptor = descriptor
        self.fitting_net = fitting_net
        self.les_model = les_model
        self.type_map = type_map
        self.ntypes = len(type_map)

        self.rcut = descriptor.get_rcut()
        self.sel = descriptor.get_sel()
        self.sel_type = list(range(self.ntypes))

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

    def fitting_output_def(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "energy",
                "size": 1,
                "category": "energy",
                "reduce": "sum",
            }
        ]

    def forward_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        self.desc = self.descriptor(extended_coord, extended_atype, nlist)[0]
        E_sr_atom = self.fitting_net(self.desc, atype)['energy']
        #print(self.desc.shape)
        #print(E_sr_atom)
        return E_sr_atom

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
        E_sr_atom = atomic_out
        #print(atomic_out)
        E_sr = E_sr_atom.sum(dim=1)

        E_lr_list = []
        for i in range(batch_size):
            coord_i = coord[i]
            cell_i = cell[i].unsqueeze(0)
            #ext_coord_i = extended_coord[i].unsqueeze(0)
            #ext_atype_i = extended_atype[i].unsqueeze(0)
            #nlist_i = nlist[i].unsqueeze(0)
            #desc_i = self.descriptor(ext_coord_i, ext_atype_i, nlist_i)[0]
            desc_i = self.desc[i]
            #print(desc_i.shape)
            #print(coord_i.shape)

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

        #print({'energy' : E_tot, 'force' : force_pred})
        return {'energy' : E_tot, 'force' : force_pred}
