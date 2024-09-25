import torch 


""" Lựa chọn tham số: \delta \in (0, 1): control forgetting rate
    c_t liên quan đến k1, k2. Chọn k1, k2 sao cho %SAM = 61%. Xấp xỉ, ta chọn k1=0.2, k2=0.4
    k1, k2: hyper-params
"""
class AOSAM(torch.optim.Optimizer):
    
    def __init__(self, params, base_optimizer, rho=0.05, lr=0.002, delta = 0.3, k1=0.2, k2=0.4, T=11314):
        defaults = dict(rho=rho, lr=lr, delta=delta, k1=k1, k2=k2, T=T)
        super(AOSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.device = device
        self.mu_t = 0.0
        self.sigma_t = 1e-10
        self.delta = delta

    def _grad_norm(self):
        norm = torch.norm(
                torch.stack([
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"] if p.grad is not None]),  
                    p=2)
        return norm

    def compute_ct(self, t):
        T = self.defaults["T"]
        k1 = self.defaults["k1"]
        k2 = self.defaults["k2"]
        
        return k1 * (t / T) + k2* (1 - t / T)

    # @torch.no_grad()
    # def first_step(self, zero_grad=False):
    #     grad_norm = self._grad_norm()

    #     # Tính mu_t, sigma_t, c_t
    #     self.mu_t = self.delta * self.mu_t + (1 - self.delta) * grad_norm.item()**2
    #     self.sigma_t = self.delta * self.sigma_t + (1 - self.delta) * ((grad_norm.item()**2) - self.mu_t)**2
    #     c_t = self.compute_ct

    #     if grad_norm.item()**2 >= (self.mu_t + c_t * self.sigma_t**0.5):
    #         for group in self.param_groups:
    #             scale = group["rho"] / (grad_norm + 1e-12)

    #             for p in group["params"]:
    #                 if p.grad is None: continue
    #                 self.state[p]["old_p"] = p.data.clone()
    #                 e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    
    #                 # Compute: w + e(w)
    #                 p.add_(e_w)                      

    #     if zero_grad: self.zero_grad()
    

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                
                # Compute: w + e(w)
                p.add_(e_w)                      

        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue

                # Get back to w from w + e(w)
                p.data = self.state[p]["old_p"] 

        # Update
        self.base_optimizer.step()               
        if zero_grad: self.zero_grad()


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







# from torch.optim._multi_tensor import SGD
# class SAMSGD(SGD):
#     def __init__(self, lr: float, rho: float):
#         super().__init__(lr)
#         self.param_groups[0]["rho"] = rho

#     @torch.no_grad()
#     def step(self, closure):
        
#         closure = torch.enable_grad()(closure)
#         loss = closure().detach()

#         for group in self.param_groups:
#             grads = []
#             params_with_grads = []

#             rho = group['rho']

#             # update lr
#             for p in group['params']:
#                 if p.grad is not None:
#                     grads.append(p.grad.clone().detach())
#                     params_with_grads.append(p)

#             grad_norm = torch.stack([g.detach().norm(2) for g in grads]).norm(2)
#             epsilon = grads  

#             torch._foreach_mul_(epsilon, rho / grad_norm)
#             torch._foreach_add_(params_with_grads, epsilon)
#             closure()
#             torch._foreach_sub_(params_with_grads, epsilon)

#         super().step()
#         return loss 



