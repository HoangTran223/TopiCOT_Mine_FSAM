import torch 


class FSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, device, rho=0.05, adaptive=False, lr=0.002, sigma=1, lmbda=0.9):
        defaults = dict(rho=rho, adaptive=adaptive, lr=lr)
        super(FSAM, self).__init__(params, defaults)

        # Thêm
        self.device = device

        self.sigma = sigma
        self.lmbda = lmbda

        self.base_optimizer = base_optimizer(self.param_groups)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)


    # Đoạn code ban đầu 
    # def _grad_norm(self):
    #     norm = torch.norm(
    #                 torch.stack([
    #                     ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
    #                     for group in self.param_groups for p in group["params"] if p.grad is not None ]),
    #                 p=2)
    #     return norm


    # @torch.no_grad()
    # def first_step(self, zero_grad=False):

    #     for group in self.param_groups:
    #         for p in group["params"]:      
    #             if p.grad is None: continue
    #             # Thử
    #             grad = p.grad

    #             # grad = p.grad.clone()
    #             if not "momentum" in self.state[p]:
    #                 self.state[p]["momentum"] = grad
    #             else:
    #                 # Compute d_t
    #                 p.grad -= self.state[p]["momentum"] * self.sigma            

    #                 # Compute m_t
    #                 self.state[p]["momentum"] = self.state[p]["momentum"] * self.lmbda + grad * (1 - self.lmbda)
            
    #     grad_norm = self._grad_norm()
    #     for group in self.param_groups:
    #         scale = group["rho"] / (grad_norm + 1e-12)

    #         for p in group["params"]:
    #             if p.grad is None: continue
    #             self.state[p]["old_p"] = p.data.clone()
    #             e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                
    #             # Compute: w + e(w)
    #             p.add_(e_w)                             

    #     if zero_grad: self.zero_grad()


    # @torch.no_grad()
    # def second_step(self, zero_grad=False):
    #     for group in self.param_groups:
    #         for p in group["params"]:
    #             if p.grad is None: continue

    #             # Get back to w from w + e(w)
    #             p.data = self.state[p]["old_p"]        
        
    #     # Update
    #     self.base_optimizer.step()                      

    #     if zero_grad: self.zero_grad()


    @torch.no_grad()
    def first_step(self, zero_grad=False, device='cuda'):
        # Khởi tạo grad_norm trên cùng device để tránh conflict CPU/GPU
        grad_norm = torch.tensor(0.0, device=device)

        # Tính momentum và grad_norm trong cùng một vòng lặp
        for group in self.param_groups:
            rho = group["rho"]
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None: 
                    continue

                grad = p.grad  # Không dùng clone() để tiết kiệm bộ nhớ và thời gian
                state = self.state[p]

                # Tính momentum
                if "momentum" not in state:
                    state["momentum"] = grad.clone()  # Khởi tạo momentum ban đầu bằng cách clone
                else:
                    p.grad.add_(state["momentum"], alpha=-self.sigma)  # Dùng in-place operation
                    state["momentum"].mul_(self.lmbda).add_(grad, alpha=1 - self.lmbda)  # In-place update cho momentum


                # Tính grad_norm đồng thời trong cùng vòng lặp
                grad_norm.add_(((torch.abs(p.to(device)) if adaptive else 1.0) * p.grad.to(device)).norm(2).pow(2))

        grad_norm = grad_norm.sqrt()  # sqrt chỉ gọi một lần

        # Tính toán scale và thực hiện update weights
        scale = rho / (grad_norm + 1e-12)  # Tránh chia 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: 
                    continue

                # Lưu trữ trạng thái cũ
                state = self.state[p]
                state["old_p"] = p.data.clone()  # Lưu lại trạng thái cũ trước khi thay đổi

                # Tính e(w) và thực hiện update weights
                e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale
                p.add_(e_w)  # In-place addition để tối ưu bộ nhớ

        # Clear gradient nếu cần
        if zero_grad:
            self.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # Khôi phục trạng thái ban đầu cho các tham số
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: 
                    continue

                # Lấy lại w từ old_p
                p.data.copy_(self.state[p]["old_p"])  # In-place copy để khôi phục trạng thái cũ

        # Gọi optimizer step để cập nhật weights
        self.base_optimizer.step()

        # Clear gradient nếu cần
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        # Closure do a full forward-backward pass
        closure = torch.enable_grad()(closure)  
        self.first_step(zero_grad=True)        
        closure()
        self.second_step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups