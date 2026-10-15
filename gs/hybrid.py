"""MCMC densification guided by absgrad's density signal.

MCMC keeps a fixed splat budget, recycles faint ("dead") splats and adds new ones by cloning
existing splats, and jitters the faintest ones while learning - a filter for useful splats.
It picks which splats to clone by opacity alone. absgrad measures where the photos' texture
pulls hardest (the sum of each pixel's absolute screen-space gradient, so opposing pulls over
busy detail don't cancel) - a density signal. This strategy is MCMC throughout, but clones
the splats that are both opaque and where that pull is strongest:

    weight = opacity x (average absolute screen-space gradient since the last refine) ^ power

Everything else (budget, recycling, noise, schedule) is gsplat's MCMCStrategy unchanged.
"""
import torch
from gsplat.strategy import MCMCStrategy
from gsplat.strategy.ops import _multinomial_sample, _update_param_with_optimizer, compute_relocation


class HybridStrategy(MCMCStrategy):
    def __init__(self, power=1.0, **kw):
        super().__init__(**kw)
        self.power = power
        self._state = None

    def initialize_state(self):
        s = super().initialize_state()
        s.update(grad=None, count=None)
        return s

    def step_pre_backward(self, params, optimizers, state, step, info):
        info["means2d"].retain_grad()

    @torch.no_grad()
    def _accumulate(self, params, state, info):
        n, dev = len(params["means"]), params["means"].device
        if state["grad"] is None or len(state["grad"]) != n:
            state["grad"] = torch.zeros(n, device=dev)
            state["count"] = torch.zeros(n, device=dev)
        m2d = info["means2d"]
        g = getattr(m2d, "absgrad", None)
        if g is None:
            g = m2d.grad.abs() if m2d.grad is not None else None
        if g is None:
            return
        g = g.clone()
        g[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        g[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        norms = g.norm(dim=-1)
        if "gaussian_ids" in info and info["gaussian_ids"] is not None and norms.dim() == 1:   # packed
            ids = info["gaussian_ids"]
            state["grad"].index_add_(0, ids, norms)
            state["count"].index_add_(0, ids, torch.ones_like(norms))
        else:   # [C, N] layout
            r = info["radii"]
            sel = (r > 0).all(-1) if r.dim() == norms.dim() + 1 else r > 0
            cam, ids = sel.nonzero(as_tuple=True)
            state["grad"].index_add_(0, ids, norms[cam, ids])
            state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=norms.dtype))

    def _weights(self, params, idx=None):
        op = torch.sigmoid(params["opacities"].flatten())
        s = self._state
        if s is None or s["grad"] is None or len(s["grad"]) != len(op):
            w = op
        else:
            avg = s["grad"] / s["count"].clamp_min(1)
            avg = avg / avg[avg > 0].mean().clamp_min(1e-12) if (avg > 0).any() else torch.ones_like(avg)
            w = op * (avg + 1e-3) ** self.power
        return w if idx is None else w[idx]

    def step_post_backward(self, params, optimizers, state, step, info, lr):
        self._accumulate(params, state, info)
        self._state = state
        refining = self.refine_start_iter < step < self.refine_stop_iter and step % self.refine_every == 0
        super().step_post_backward(params, optimizers, state, step, info, lr)
        if refining:   # start a fresh gradient window for the new splat set
            n, dev = len(params["means"]), params["means"].device
            state["grad"] = torch.zeros(n, device=dev)
            state["count"] = torch.zeros(n, device=dev)

    @torch.no_grad()
    def _relocate_gs(self, params, optimizers, binoms):
        opacities = torch.sigmoid(params["opacities"])
        dead = opacities.flatten() <= self.min_opacity
        n = int(dead.sum())
        if n == 0:
            return 0
        dead_idx = dead.nonzero(as_tuple=True)[0]
        alive_idx = (~dead).nonzero(as_tuple=True)[0]
        sampled = alive_idx[_multinomial_sample(self._weights(params, alive_idx), n, replacement=True)]
        self._clone(params, optimizers, binoms, sampled, into=dead_idx)
        return n

    @torch.no_grad()
    def _add_new_gs(self, params, optimizers, binoms):
        cur = len(params["means"])
        n = max(0, min(self.cap_max, int(1.05 * cur)) - cur)
        if n == 0:
            return 0
        sampled = _multinomial_sample(self._weights(params), n, replacement=True)
        self._clone(params, optimizers, binoms, sampled, into=None)
        return n

    def _clone(self, params, optimizers, binoms, sampled, into):
        """MCMC's relocate (into = dead indices) / sample_add (into = None), sources given."""
        eps = torch.finfo(torch.float32).eps
        opacities = torch.sigmoid(params["opacities"])
        new_op, new_sc = compute_relocation(
            opacities=opacities[sampled], scales=torch.exp(params["scales"])[sampled],
            ratios=torch.bincount(sampled)[sampled] + 1, binoms=binoms)
        new_op = torch.clamp(new_op, max=1.0 - eps, min=self.min_opacity)

        def param_fn(name, p):
            if name == "opacities":
                p[sampled] = torch.logit(new_op)
            elif name == "scales":
                p[sampled] = torch.log(new_sc)
            if into is not None:
                p[into] = p[sampled]
                return torch.nn.Parameter(p, requires_grad=p.requires_grad)
            return torch.nn.Parameter(torch.cat([p, p[sampled]]), requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            if into is not None:
                v[sampled] = 0
                return v
            return torch.cat([v, torch.zeros((len(sampled), *v.shape[1:]), device=v.device)])

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
