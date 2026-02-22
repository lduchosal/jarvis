# POC v4 — PlayerModel Training (Python)

## Objectif

Entrainer un agent RL (policy network) a jouer au Pokemon TCG via **REINFORCE**. Le training se fait en Python (PyTorch), dans le meme dossier `training/` que le world model.

L'environnement de training est le **world model PyTorch** lui-meme — pas le moteur de regles C#. La policy "reve" a l'interieur du world model appris.

## Pipeline

```
data/transitions/*.jsonl    →  etats initiaux (obs reelles)
data/wm-checkpoints/best.pt →  WorldModelNet (env de training)
                                     │
                              PolicyNet joue dans le WM
                              REINFORCE optimise la policy
                                     │
                              checkpoints/policy_*.pt
                                     │
                              export_policy_onnx.py
                                     │
                              models/policy.onnx → C# ModelBot
```

## Pourquoi le world model comme env

Le world model (`WorldModelNet`, MLP 128→256→256→4 tetes) predit :

```
(obs[122], action_onehot[6]) → (delta_obs[122], reward[1], done_logit[1], delta_hp[2])
```

C'est exactement l'interface d'un `env.step()` :
- `next_obs = obs + delta_obs` (avec correction HP via `delta_hp`)
- `reward` directement predit
- `done = sigmoid(done_logit) > 0.5`

Avantages :
- **Zero rewrite** du moteur C# — le WM est deja un simulateur
- **Rapide** — un forward pass PyTorch, pas d'IPC cross-language
- **Differentiable** — on peut backprop a travers le WM si besoin (hors scope v4)
- **Deja valide** — le benchmark v3.10 passe sur 200K games
- **GPU-native** — tout reste en tenseurs GPU, zero aller-retour CPU

Risque :
- **Model bias** — la policy apprend les erreurs du WM, pas la realite
- Mitigation : valider la policy finale sur le vrai moteur C# via `bench-policy`

## Architecture GPU-only : rollouts batches

Le design cle : **B episodes en parallele, tout en tenseurs sur le meme device (GPU/MPS)**. Pas un seul `.item()`, `.numpy()`, ou `list[int]` dans la boucle de rollout.

### Principe

```
PolicyNet (sur GPU, avec grad)  ←→  WorldModelNet (sur GPU, sans grad, frozen)
          ↕                                    ↕
     obs [B, 122]  ────────────────→  inp [B, 128]
     logits [B, 6]                    delta_obs [B, 122]
     actions [B]                      reward [B]
     log_probs [B]                    done [B]
```

Les deux modeles vivent sur GPU. L'obs circule comme tenseur GPU entre les deux sans jamais toucher le CPU.

### Masque d'actions legales (pur tensor)

Derive directement des features de l'observation — aucun port du moteur C# :

```python
# training/wm_env.py

def legal_mask_from_obs(obs: torch.Tensor) -> torch.Tensor:
    """
    Derive le masque d'actions legales directement depuis obs[B, 122].
    Retourne [B, 6] avec 0.0 si legal, -inf si illegal.
    Tout en ops tensorielles, zero CPU.
    """
    B = obs.size(0)
    mask = torch.zeros(B, ACTION_DIM, device=obs.device)

    # Pass (action 0) : toujours legal → mask[:, 0] = 0 deja

    # AttachEnergy (action 1) : illegal si deja attache ce tour
    mask[:, 1] = torch.where(obs[:, 59] > 0.5, float("-inf"), 0.0)

    # Attack0-3 (actions 2-5) : illegal si attaque n'existe pas OU energie insuffisante
    total_energy = obs[:, 48:59].sum(dim=1)  # [B] somme des 11 types d'energie
    for i in range(4):
        atk_missing = obs[:, 36 + i] < 0.5               # atkExists[i] == 0
        cost = obs[:, 44 + i] * 5.0                       # atkTotalCost[i] denormalize
        too_costly = total_energy < cost
        mask[:, 2 + i] = torch.where(atk_missing | too_costly, float("-inf"), 0.0)

    return mask
```

Features utilisees (depuis `feature_layout.py`) :
- `obs[:, 59]` = `self.energyAttached` (0/1)
- `obs[:, 48:59]` = `self.energy[type]` (11 types, raw counts)
- `obs[:, 36:40]` = `self.atkExists[0-3]` (0/1)
- `obs[:, 44:48]` = `self.atkTotalCost[0-3]` (normalise /5)

### Rollout batche

```python
# training/train_policy.py

def batched_rollout(policy, wm, initial_obs_pool, B, max_steps, device):
    """
    Joue B episodes en parallele dans le world model.
    Tout sur GPU, zero conversion CPU dans la boucle.

    Args:
        policy: PolicyNet (sur device, mode train)
        wm: WorldModelNet (sur device, mode eval, frozen)
        initial_obs_pool: [N, 122] tensor sur device
        B: nombre d'episodes en parallele
        max_steps: limite de steps par episode
        device: torch.device

    Returns:
        log_probs: [max_steps, B] — log π(a|s) pour le policy gradient
        rewards: [max_steps, B] — reward a chaque step
        dones: [max_steps, B] — flag done (bool)
        lengths: [B] — duree de chaque episode
    """
    # Echantillonner B observations initiales
    idx = torch.randint(len(initial_obs_pool), (B,), device=device)
    obs = initial_obs_pool[idx].clone()  # [B, 122]

    # Buffers pre-alloues
    all_log_probs = torch.zeros(max_steps, B, device=device)
    all_rewards = torch.zeros(max_steps, B, device=device)
    all_dones = torch.zeros(max_steps, B, dtype=torch.bool, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)  # episodes en cours
    lengths = torch.zeros(B, dtype=torch.long, device=device)

    for t in range(max_steps):
        if not active.any():
            break

        # 1. Policy forward (avec grad pour log_prob)
        logits = policy(obs)                          # [B, 6]
        mask = legal_mask_from_obs(obs)                # [B, 6]
        dist = torch.distributions.Categorical(logits=logits + mask)
        actions = dist.sample()                        # [B]
        log_probs = dist.log_prob(actions)             # [B]

        # 2. WM forward (sans grad — env frozen)
        with torch.no_grad():
            action_oh = F.one_hot(actions, ACTION_DIM).float()     # [B, 6]
            wm_input = torch.cat([obs, action_oh], dim=1)          # [B, 128]
            delta_obs, reward, done_logit, delta_hp = wm(wm_input)

            # 3. Appliquer le delta
            next_obs = obs + delta_obs.squeeze()

            # HP via tete dediee (plus precis que delta_obs pour les HP)
            next_obs[:, SELF_HP_IDX] = obs[:, SELF_HP_IDX] + delta_hp[:, 0]
            next_obs[:, OPP_HP_IDX]  = obs[:, OPP_HP_IDX]  + delta_hp[:, 1]

            # Clamp HP dans [0, maxHP]
            next_obs[:, SELF_HP_IDX] = next_obs[:, SELF_HP_IDX].clamp(
                min=0, max=obs[:, SELF_HP_IDX + 1])  # maxHP = index 12
            next_obs[:, OPP_HP_IDX] = next_obs[:, OPP_HP_IDX].clamp(
                min=0, max=obs[:, OPP_HP_IDX + 1])   # maxHP = index 72

            # 4. Done
            done = (torch.sigmoid(done_logit.squeeze(-1)) > 0.5) | \
                   (next_obs[:, OPP_HP_IDX] <= 0)

            # Forcer reward +1 sur KO
            r = reward.squeeze(-1)
            ko = next_obs[:, OPP_HP_IDX] <= 0
            r = torch.where(ko & done, torch.maximum(r, torch.ones_like(r)), r)

        # 5. Stocker (seulement les episodes actifs)
        all_log_probs[t] = torch.where(active, log_probs, torch.zeros_like(log_probs))
        all_rewards[t] = torch.where(active, r, torch.zeros_like(r))
        all_dones[t] = done & active
        lengths += active.long()

        # 6. Desactiver les episodes termines
        newly_done = done & active
        active = active & ~done
        obs = next_obs

    return all_log_probs, all_rewards, all_dones, lengths
```

### REINFORCE sur le batch

```python
def train_batch(policy, optimizer, wm, initial_obs_pool, B, max_steps, device,
                gamma=0.99, baseline=0.0, baseline_decay=0.99):
    """
    1 iteration = B episodes en parallele → 1 policy gradient update.
    """
    log_probs, rewards, dones, lengths = batched_rollout(
        policy, wm, initial_obs_pool, B, max_steps, device)

    # Discounted returns (backward pass sur le temps)
    T = max_steps
    returns = torch.zeros(T, B, device=device)
    G = torch.zeros(B, device=device)
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G * (~dones[t]).float()
        returns[t] = G

    # Masque temporel (ignorer les steps apres done)
    step_idx = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
    valid = step_idx < lengths.unsqueeze(0)                   # [T, B]

    # Baseline (moyenne des returns initiaux)
    episode_returns = returns[0]                               # [B]
    baseline = baseline_decay * baseline + (1 - baseline_decay) * episode_returns.mean().item()

    # Policy gradient loss
    advantages = returns - baseline                            # [T, B]
    pg_loss = -(log_probs * advantages.detach() * valid.float()).sum() / valid.float().sum()

    optimizer.zero_grad()
    pg_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": pg_loss.item(),
        "avg_return": episode_returns.mean().item(),
        "avg_length": lengths.float().mean().item(),
        "win_rate": (episode_returns > 0.5).float().mean().item(),
        "baseline": baseline,
    }
```

### Performance attendue

| Aspect | 1 episode CPU (ancienne spec) | B=256 GPU batch |
|---|---|---|
| Forward passes / iteration | ~15 (1 episode) | ~15 × 256 = 3840 |
| Transferts GPU↔CPU / step | 2 (obs + action) | 0 |
| Utilisation GPU | ~5% (kernels trop petits) | ~80%+ (batches larges) |
| Episodes / seconde (estime) | ~500 | ~50K+ |

Le bottleneck passe du transfert memoire au compute pur — ce qu'on veut.

### Etats initiaux

On extrait les observations de tour 0 depuis le dataset de transitions existant :

```python
def load_initial_obs(jsonl_path: str, device: torch.device) -> torch.Tensor:
    """Extraire les obs du premier step de chaque game. Tensor sur device."""
    obs_list = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            if row["obs"][121] < 0.02:  # turnIndex/50 < 0.02 → tour 0
                obs_list.append(row["obs"])
    return torch.tensor(obs_list, dtype=torch.float32, device=device)
```

Charge une fois, stocke sur GPU. Echantillonne par `torch.randint` sans copie.

## Architecture du reseau (PolicyNet)

```
Input:  obs[122]
             |
       Linear(122 → 256) + ReLU
             |
       Linear(256 → 256) + ReLU
             |
       Linear(256 → 6)          logits bruts
```

~70K parametres.

```python
# training/policy_model.py
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACTION_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # logits [B, 6]
```

## Algorithme : REINFORCE (Williams, 1992)

REINFORCE est l'algorithme de **policy gradient** le plus simple. C'est une implementation specifique de RL, pas un synonyme (voir `doc/faq/reinforcement_learning.md`).

Principe : jouer un episode complet, puis ajuster la policy proportionnellement au return obtenu :

```
∇J(θ) = E[ Σ_t  G_t · ∇ log π_θ(a_t | s_t) ]
```

- Si le return G_t etait bon → renforcer l'action prise (augmenter sa probabilite)
- Si le return etait mauvais → decourager l'action (diminuer sa probabilite)

On ajoute une **baseline** (moyenne mobile des returns) pour reduire la variance :

```
∇J(θ) = E[ Σ_t  (G_t - b) · ∇ log π_θ(a_t | s_t) ]
```

### Pourquoi REINFORCE et pas PPO

| | REINFORCE | PPO |
|---|---|---|
| Complexite | ~30 loc | ~150 loc (clipping, value net, GAE) |
| Reseau | Policy seul | Policy + Value network |
| Variance | Haute (baseline aide) | Basse (GAE + clipping) |
| Stabilite | Peut diverger | Tres stable |

REINFORCE suffit ici car : 6 actions, episodes courts (~15 steps), signal clair (+1/-1). Si REINFORCE ne converge pas bien, **PPO est l'upgrade naturel en v5** — il ne change que le training loop, pas l'architecture du PolicyNet ni le rollout batche.

## Hyperparametres

| Parametre | Valeur | Justification |
|---|---|---|
| `B` (batch episodes) | 256 | Maximise le throughput GPU, assez pour du variance reduction |
| `max_steps` | 50 | Eviter les rollouts infinis dans le WM |
| `gamma` | 0.99 | Matchs courts (~10-20 tours) |
| `lr` | 3e-4 | Standard Adam pour policy gradient |
| `grad_clip` | 1.0 | Meme valeur que le world model |
| `baseline_decay` | 0.99 | EMA lissee |
| `hidden_dim` | 256 | Coherent avec le world model |
| `iterations` | 2000 | 2000 × 256 = ~500K episodes total |

## Recompenses

| Signal | Valeur | Source |
|---|---|---|
| Victoire | +1.0 | `done=true` et `opp_HP <= 0` |
| Defaite | -1.0 | `done=true` et `self_HP <= 0` |
| Illegal | ~-0.01 | WM predit reward ~ -0.01 pour les actions illegales |
| Intermediate | ~0.0 | WM predit le reward appris des donnees |

## Regimes d'entrainement

### Phase 1 — Policy dans le WM

```
PolicyNet joue dans WorldModelNet (pas d'adversaire explicite)
Le WM simule le match complet du point de vue du joueur courant
Objectif : return moyen > 0.5, win_rate > 0.7 dans le WM
Iterations : ~2000 (= ~500K episodes avec B=256)
```

Note : le WM est entraine sur des transitions du point de vue du joueur qui agit. La policy joue donc contre un "adversaire implicite" encode dans la dynamique du WM.

### Phase 2 — Validation sur le vrai moteur

```
Charger policy.onnx dans C# ModelBot
bench-policy --p1 modelbot --p2 random → win rate > 70%
bench-policy --p1 modelbot --p2 attackbot → win rate > 55%
```

Si la policy entraine dans le WM ne transfere pas bien au vrai moteur → le WM a un biais trop fort, il faut ameliorer le WM d'abord.

## Metriques et logging

```python
# Toutes les 10 iterations (= 2560 episodes)
metrics = {
    "iteration":  i,
    "episodes":   i * B,
    "win_rate":   fraction des episodes avec return > 0.5,
    "avg_return": moyenne des episode returns,
    "avg_length": duree moyenne en steps,
    "entropy":    entropie moyenne de la policy,
    "loss":       policy gradient loss,
}
```

Support wandb (meme pattern que `train.py`).

## Export ONNX

```python
# training/export_policy_onnx.py
def export(checkpoint_path, output_path="models/policy.onnx"):
    net = PolicyNet()
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    net.eval()

    dummy = torch.randn(1, OBS_DIM)
    torch.onnx.export(
        net, dummy, output_path,
        input_names=["obs"],
        output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
    )
```

### Format ONNX

```
Input:  "obs"     float32[B, 122]
Output: "logits"  float32[B, 6]
```

Pas de softmax dans le modele ONNX — le masquage et softmax se font cote C#.

## Structure de fichiers

Ajouts dans le dossier `training/` existant :

```
training/
  model.py                  # WorldModelNet (existe)
  dataset.py                # TransitionDataset (existe)
  feature_layout.py         # Constantes OBS/ACTION (existe)
  train.py                  # Training WM (existe)
  export_onnx.py            # Export WM (existe)
  policy_model.py           # PolicyNet (nouveau)
  wm_env.py                 # legal_mask_from_obs, load_initial_obs (nouveau)
  train_policy.py           # batched_rollout + REINFORCE (nouveau)
  export_policy_onnx.py     # Export policy → ONNX (nouveau)
```

## CLI

```bash
# Training policy dans le world model
python training/train_policy.py \
  --wm-checkpoint data/wm-checkpoints/best.pt \
  --transitions data/transitions.jsonl \
  --batch 256 --iterations 2000 --lr 3e-4 \
  --output data/checkpoints/

# Export ONNX
python training/export_policy_onnx.py \
  --checkpoint data/checkpoints/policy_best.pt \
  --output models/policy.onnx

# Validation sur le vrai moteur C#
dotnet run --project src/Ptcgo2.Console -- bench-policy \
  --p1 modelbot --p1-checkpoint models/policy.onnx \
  --p2 random --games 10000
```

## Criteres de succes

- Win rate > 0.7 dans le WM apres convergence
- Win rate > 70% vs Random sur le vrai moteur C# (transfert WM → realite)
- Win rate > 55% vs AttackBot sur le vrai moteur C#
- Export ONNX fonctionne (memes logits PyTorch vs ONNX, tolerance 1e-5)
- Zero transfert GPU↔CPU dans la boucle de rollout

## Limitations (hors scope v4)

- Pas de backprop a travers le WM (REINFORCE est gradient-free cote env)
- Pas de self-play (un seul joueur, adversaire implicite dans le WM)
- Pas de PPO (REINFORCE suffit pour 6 actions)
- Pas de fine-tuning sur le vrai moteur (policy entierement apprise dans le WM)
- L'adversaire implicite est le "joueur random moyen" (donnees de training du WM)

## Auteurs

Q Humain, le 2026-02-19
Claude Opus 4.6 (Anthropic), redaction, le 2026-02-19
