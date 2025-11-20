import torch
import torch.nn as nn
import torch.nn.functional as F

class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, seq_len=8, simnorm_dim=8):
        super().__init__()
        self.L = seq_len
        self.dim = simnorm_dim

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        shp = x.shape
        x = x.view(*shp[:-1], self.L, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"Simplicial Embeddings(dim={self.dim})"


class GumbelST(nn.Module):
    def __init__(self, V=8):
        super().__init__()
        self.V = V

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.V)
        x = F.gumbel_softmax(x, hard=True, dim=-1)
        return x.view(*shp)


class Norm(nn.Module):
    def __init__(self, p=2, beta=1.):
        super().__init__()
        self.p = p
        self.beta = beta

    def forward(self, x):
        diff = self.beta*torch.norm(x, p=self.p, dim=-1).mean()
        return x, diff, None


class VectorQuantization(nn.Module):
    """
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937

    Built off Karpathy's implementation
    https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/model/quantize.py

    Follows the original DeepMind implementation
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
    """
    def __init__(self, V, embedding_dim=64, kld_scale=10., commitment_cost=0.25):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.V = V

        self.kld_scale = kld_scale
        self.commitment_cost = commitment_cost

        # self.embed = nn.Parameter(L, V, embedding_dim)
        self.embed = nn.Parameter(torch.empty(V, embedding_dim))
        nn.init.normal_(self.embed)

        self.register_buffer('data_initialized', torch.zeros(1))

    def forward(self, z_e):
        B, D = z_e.size()

        z_e = z_e.reshape(B, -1, self.embedding_dim)
        # TODO, Make this LxV. Currently this is only a long V vecetor with L=1

        dist = (
            z_e.pow(2).sum(-1, keepdim=True)
            - 2 * z_e @ self.embed.t()
            + self.embed.pow(2).sum(-1, keepdim=True).t()
        )
        _, ind = (-dist).max(-1)
        ind = ind.view(B, -1)

        # vector quantization cost that trains the embedding vectors
        z_q = self.embed_code(ind) # (B, H, W, C)
        diff = self.commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()
        diff *= self.kld_scale

        z_q = z_e + (z_q - z_e).detach() # noop in forward pass, straight-through gradient estimator in backward pass
        z_q = z_q.view(B, -1) # stack encodings into channels again: (B, C, H, W)
        return z_q, diff, ind

    def embed_code(self, embed_id):
        # TODO Fix this, perhaps with one-hot
        one_hot_id = F.one_hot(embed_id, self.V).to(self.embed.dtype)
        return one_hot_id @ self.embed


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), learnable_ln=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features, elementwise_affine=learnable_ln)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


class CRELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, x_ = torch.relu(x), torch.relu(-x)
        return torch.cat((x, x_), 1)


class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        sim_type: str,
        sim_dimension: int,
        seq_len: int,
        norm_beta: float,
        device: torch.device = None,
    ):
        super().__init__()
        if sim_type in ["vq_both", "vq_critic"]:
            self.act_fn=VectorQuantization(sim_dimension)
        elif sim_type in ["gst_both", "gst_critic"]:
            self.act_fn=GumbelST(sim_dimension)
        elif sim_type in ["sim_both, sim_critic"]:
            self.act_fn=SimNorm(seq_len, sim_dimension)
        elif sim_type in ["l1_both", "l1_critic"]:
            self.act_fn=Norm(1, beta=norm_beta)
        elif sim_type in ["l2_both", "l2_critic"]:
            self.act_fn=Norm(2, beta=norm_beta)
        elif sim_type == "crelu_critic":
            self.act_fn = CRELU()
        else:
            self.act_fn=None
        self.act_type=sim_type
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        if (self.act_type in ["sim_both", "sim_critic", "gst_both", "gst_critic"]):
            self.fc_head = nn.Sequential(
                                NormedLinear(hidden_dim // 2, seq_len*sim_dimension, act=self.act_fn),
                                nn.Linear(seq_len*sim_dimension, num_atoms),
                            )
        elif self.act_type in ["vq_both", "vq_critic"]:
            self.fc_head = NormedLinear(hidden_dim // 2, hidden_dim // 4, act=self.act_fn)
        elif self.act_type in ["l1_both", "l1_actor", "l2_both", "l2_actor"]:
            self.fc_head = NormedLinear(hidden_dim // 2, hidden_dim // 4, act=self.act_fn, learnable_ln=True)
        elif self.act_type == "crelu_critic":
            self.fc_head = nn.Sequential(
                        nn.Linear(hidden_dim // 2, hidden_dim // 4),
                        CRELU(),
                        nn.Linear(hidden_dim // 2, num_atoms),
                    )
        else:
            self.fc_head =  nn.Sequential(
                                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                                nn.ReLU(),
                                nn.Linear(hidden_dim // 4, num_atoms),
                            )

        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.hidden_dim = hidden_dim

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        x = self.fc_head(x)
        return x

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = (
            rewards.unsqueeze(1)
            + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        )
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        l_mask = torch.logical_and((u > 0), (l == u))
        u_mask = torch.logical_and((l < (self.num_atoms - 1)), (l == u))

        l = torch.where(l_mask, l - 1, l)
        u = torch.where(u_mask, u + 1, u)

        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )
        return proj_dist


class DistributionalQNetworkTuple(DistributionalQNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc_head_out = nn.Linear(self.hidden_dim // 4, self.num_atoms)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        x, diff, id = self.fc_head(x)
        x = self.fc_head_out(x)
        return x, diff, id

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = (
            rewards.unsqueeze(1)
            + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        )
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        l_mask = torch.logical_and((u > 0), (l == u))
        u_mask = torch.logical_and((l < (self.num_atoms - 1)), (l == u))

        l = torch.where(l_mask, l - 1, l)
        u = torch.where(u_mask, u + 1, u)

        next_dist = F.softmax(self.forward(obs, actions)[0], dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )
        return proj_dist


class Critic(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        sim_type: str,
        sim_dimension: int,
        seq_len: int,
        norm_beta: float,
        device: torch.device = None,
    ):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            sim_type=sim_type,
            sim_dimension=sim_dimension,
            seq_len=seq_len,
            norm_beta=norm_beta,
            device=device,
        )
        self.qnet2 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            sim_type=sim_type,
            sim_dimension=sim_dimension,
            seq_len=seq_len,
            norm_beta=norm_beta,
            device=device,
        )

        self.register_buffer(
            "q_support", torch.linspace(v_min, v_max, num_atoms, device=device)
        )
        self.device = device

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
    ) -> torch.Tensor:
        """Projection operation that includes q_support directly"""
        q1_proj = self.qnet1.projection(
            obs,
            actions,
            rewards,
            bootstrap,
            discount,
            self.q_support,
            self.q_support.device,
        )
        q2_proj = self.qnet2.projection(
            obs,
            actions,
            rewards,
            bootstrap,
            discount,
            self.q_support,
            self.q_support.device,
        )
        return q1_proj, q2_proj

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        return torch.sum(probs * self.q_support, dim=1)


class Actor(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        std_min: float = 0.05,
        std_max: float = 0.8,
        sim_type: str = "",
        sim_dimension: int = 64,
        seq_len: int=8,
        norm_beta: float=1.,
        device: torch.device = None,
    ):
        super().__init__()
        if sim_type in ["vq_both", "vq_actor"]:
            self.act_fn=VectorQuantization(sim_dimension)
        elif sim_type in ["gst_both", "gst_actor"]:
            self.act_fn=GumbelST(sim_dimension)
        elif sim_type in ["sim_both", "sim_actor"]:
            self.act_fn=SimNorm(seq_len, sim_dimension)
        elif sim_type in ["l1_both", "l1_actor"]:
            self.act_fn=Norm(1, beta=norm_beta)
        elif sim_type in ["l2_both", "l2_actor"]:
            self.act_fn=Norm(2, beta=norm_beta)
        elif sim_type == "crelu_actor":
            self.act_fn = CRELU()
        else:
            self.act_fn=None
        self.act_type = sim_type
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        if (self.act_type in ["sim_both", "sim_actor", "vq_both", "vq_actor", "gst_both", "gst_actor", "global_simnorm_actor"]):
            self.fc_head = NormedLinear(hidden_dim // 2, seq_len*sim_dimension, act=self.act_fn)
            self.fc_mu = nn.Sequential(
                nn.Linear(seq_len*sim_dimension, n_act),
                nn.Tanh(),
            )
        elif self.act_type in ["l1_both", "l1_actor", "l2_both", "l2_actor"]:
            # Cannot learn affine params of LN otherwise activation can be arbirarily small learning to useless L1/L2 norm.
            self.fc_head = NormedLinear(hidden_dim // 2, hidden_dim // 4, act=self.act_fn, learnable_ln=True)
            self.fc_mu = nn.Sequential(
                nn.Linear(hidden_dim // 4, n_act),
                nn.Tanh(),
            )
        elif self.act_type == "crelu_actor":
            self.fc_head = nn.Sequential(
                        nn.Linear(hidden_dim // 2, hidden_dim // 4),
                        CRELU(),
                    )
            self.fc_mu = nn.Sequential(
                nn.Linear(hidden_dim // 2, n_act),
                nn.Tanh(),
            )
        else:
            self.fc_head =  nn.Sequential(
                                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                                nn.ReLU(),
                            )
            self.fc_mu = nn.Sequential(
                nn.Linear(hidden_dim // 4, n_act),
                nn.Tanh(),
            )


        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

        noise_scales = (
            torch.rand(num_envs, 1) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)

        self.register_buffer("std_min", torch.as_tensor(std_min))
        self.register_buffer("std_max", torch.as_tensor(std_max))
        self.n_envs = num_envs
        self.device = device

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        x = self.net(x)
        x = self.fc_head(x)
        action = self.fc_mu(x)
        return action

    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
    ) -> torch.Tensor:
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments (one per environment)
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(
                torch.where(dones_view, new_scales, self.noise_scales)
            )

        act = self(obs)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
        return act + noise


class TupleActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        x = self.net(x)
        x, diff, ind = self.fc_head(x)
        action = self.fc_mu(x)
        return action, diff, ind

    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
    ) -> torch.Tensor:
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments (one per environment)
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(
                torch.where(dones_view, new_scales, self.noise_scales)
            )

        act = self(obs)[0]
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
        return act + noise


class TupleCritic(Critic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qnet1 = DistributionalQNetworkTuple(*args, **kwargs)
        self.qnet2 = DistributionalQNetworkTuple(*args, **kwargs)


class MultiTaskActor(Actor):
    def __init__(self, num_tasks: int, task_embedding_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.task_embedding_dim = task_embedding_dim
        self.task_embedding = nn.Embedding(
            num_tasks, task_embedding_dim, max_norm=1.0, device=self.device
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: Optimize the code to be compatible with cudagraphs
        # Currently in-place creation of task_indices is not compatible with cudagraphs
        task_ids_one_hot = obs[..., -self.num_tasks :]
        task_indices = torch.argmax(task_ids_one_hot, dim=1)
        task_embeddings = self.task_embedding(task_indices)
        obs = torch.cat([obs[..., : -self.num_tasks], task_embeddings], dim=-1)
        return super().forward(obs)


class MultiTaskCritic(Critic):
    def __init__(self, num_tasks: int, task_embedding_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.task_embedding_dim = task_embedding_dim
        self.task_embedding = nn.Embedding(
            num_tasks, task_embedding_dim, max_norm=1.0, device=self.device
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # TODO: Optimize the code to be compatible with cudagraphs
        # Currently in-place creation of task_indices is not compatible with cudagraphs
        task_ids_one_hot = obs[..., -self.num_tasks :]
        task_indices = torch.argmax(task_ids_one_hot, dim=1)
        task_embeddings = self.task_embedding(task_indices)
        obs = torch.cat([obs[..., : -self.num_tasks], task_embeddings], dim=-1)
        return super().forward(obs, actions)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
    ) -> torch.Tensor:
        task_ids_one_hot = obs[..., -self.num_tasks :]
        task_indices = torch.argmax(task_ids_one_hot, dim=1)
        task_embeddings = self.task_embedding(task_indices)
        obs = torch.cat([obs[..., : -self.num_tasks], task_embeddings], dim=-1)
        return super().projection(obs, actions, rewards, bootstrap, discount)
