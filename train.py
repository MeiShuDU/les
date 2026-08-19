import torch
from deepmd.pt.loss.ener import EnergyStdLoss
from deepmd.dpmodel.utils.learning_rate import LearningRateExp

# ... 假设你已经实例化了 model (HybridLESAtomicModel)
# ... 并且准备好了 train_loader 和 valid_loader
device = torch.device('cpu')

def Train(
        model,
        train_loader,
        device

):
    
    loss_func = EnergyStdLoss(
        start_pref_e=0.02,  # 初始能量损失权重
        limit_pref_e=1.0,   # 最终能量损失权重
        start_pref_f=1000.0,# 初始力损失权重（通常较大，因为力的数值大且原子多）
        limit_pref_f=1.0,   # 最终力损失权重
        relative_f=0.1,     # 可选：使用相对力误差
    )
    # 2. 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. 定义学习率调度器
    lr_scheduler = LearningRateExp(
        start_lr=1e-3,
        stop_lr=3.51e-8,
        decay_steps=5000,
        stop_steps=100000
    )

# 4. 训练循环
    global_step = 0
    n_epochs = 100
    for _ in range(n_epochs):
        for batch in train_loader:
            input_dict = {
                'coord': batch['coord'].to(device),
                'atype': batch['atype'].to(device),
                'cell': batch['cell'].to(device),
            }
            label = {
                'energy': batch['energy'].to(device),
                'force': batch['force'].to(device),
                'find_energy': 1.0,
                'find_force': 1.0,
            }
            natoms = input_dict['coord'].shape[1]

            lr = lr_scheduler.value(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            model_pred, loss, more_loss = loss_func(
                input_dict, model, label, natoms, lr, mae=False
            )

            print(loss)

        # --- 反向传播与优化 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step >= lr_scheduler.stop_steps : return

# 1. 定义损失函数


