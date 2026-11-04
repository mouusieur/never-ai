# Spécification Mathématique & Architecturale : `ChronoMorph-LM`

Ce document formalise les équations exactes, les dimensions tensorielles et les théorèmes d'invariance implémentés dans le cœur neuronal de **`never-ai`**.

---

## 1. Flux Tensoriel d'un Bloc `ChronoMorph`

Soit une séquence d'entrée $X \in \mathbb{N}^{B \times T}$ de vocabulaire $\mathcal{V}$.

### 1.1 Embeddings Découplés & Connexions U-Net
1. **Embedding de tokens & Value Embeddings :**
   $$h^{(0)} = E_{\text{tok}}[X] \in \mathbb{R}^{B \times T \times d}$$
   Les embeddings d'entrée $E_{\text{tok}}$ et la tête de sortie $W_{\text{head}} \in \mathbb{R}^{|\mathcal{V}| \times d}$ sont découplés (*untied*, cf. *Modded-NanoGPT*) afin de permettre une optimisation différenciée et d'éviter le couplage parasite lors des mues de vocabulaire.
2. **Connexions Résiduelles Symétriques U-Net :**
   Pour un réseau de profondeur $L$, la première moitié des blocs ($\ell \in \{1, \dots, \lfloor L/2 \rfloor\}$) empile ses états résiduels dans une pile $\{s_1, \dots, s_{\lfloor L/2 \rfloor}\}$. La seconde moitié réinjecte ces états avec un scalaire apprenable $\lambda_\ell$ (initialisé à $0$ lors des mues de profondeur pour préserver l'invariance) :
   $$\tilde{h}^{(\ell-1)} = h^{(\ell-1)} + \lambda_\ell \, s_{L - \ell + 1}$$

### 1.2 Multi-Head Latent Attention (MLA) + QK-RMSNorm + RoPE Découplé
Pour réduire l'empreinte mémoire du cache KV de >80 % sur notre budget de 3.8 Gio RAM tout en préservant l'expressivité multi-têtes :
1. **Compression conjointe basse-dimension KV ($d_c \ll n_h d_h$, typiquement $d_c = d / 4$) :**
   $$c_t^{KV} = \text{RMSNorm}\big(x_t W^{DKV}\big) \in \mathbb{R}^{d_c}$$
   $$K_t^C = c_t^{KV} W^{UK}, \qquad V_t = c_t^{KV} W^{UV}$$
2. **Clés/Requêtes Positionnelles Découplées (RoPE / YaRN) :**
   $$q_t^R = \text{RoPE}\big(\text{RMSNorm}_Q(x_t W^{QR})\big), \qquad k_t^R = \text{RoPE}\big(\text{RMSNorm}_K(x_t W^{KR})\big)$$
   $$Q_t = [Q_t^C \;\|\; q_t^R], \qquad K_t = [K_t^C \;\|\; k_t^R]$$
   Seul le vecteur latent compressé $(c_t^{KV}, k_t^R)$ est stocké dans le cache KV d'inférence !

### 1.3 Module de Mémoire Neuro-Plastique à Long Terme (`Titans-MAG`)
En parallèle de l'attention locale (mémoire à court terme précise), une branche **Neural Long-Term Memory** $\mathcal{M}_{t}$ (MLP résiduel à 2 couches de poids rapides $\Phi_t$) mémorise le contexte lointain *en temps de test* :
1. **Projection associative :** $k_t^{\text{mem}} = x_t W_K^{\text{mem}}, \; v_t^{\text{mem}} = x_t W_V^{\text{mem}}, \; q_t^{\text{mem}} = x_t W_Q^{\text{mem}}$.
2. **Perte associative de surprise :**
   $$\ell(\Phi_{t-1}; x_t) = \left\| \mathcal{M}_{\Phi_{t-1}}(k_t^{\text{mem}}) - v_t^{\text{mem}} \right\|_2^2$$
3. **Mise à jour en ligne dépendante des données (Momentum de surprise + Oubli adaptatif) :**
   $$S_t = \eta_t S_{t-1} - \theta_t \nabla_{\Phi} \ell(\Phi_{t-1}; x_t)$$
   $$\Phi_t = (1 - \alpha_t) \Phi_{t-1} + S_t$$
   où $\alpha_t \in [0,1]$ (oubli), $\eta_t \in [0,1]$ (inertie de surprise) et $\theta_t > 0$ (plasticité instantanée) sont produits par des portes sigmoïdes légères sur $x_t$. L'évaluation est parallélisée par segments (*chunkwise parallel scan*) durant l'entraînement !
4. **Fusion Gated (`MAG` — Memory As Gate) :**
   $$y_t^{\text{mix}} = y_t^{\text{attn}} \odot \sigma\big(\mathcal{M}_{\Phi_t}(q_t^{\text{mem}})\big)$$

### 1.4 Couche FFN $\text{ReLU}^2$ / `DeepSeekMoE` Sans Perte Auxiliaire
- **Stades Denses (0–1) :**
  $$\text{FFN}(u) = \big(\max(0, u W_{\text{up}})\big)^2 W_{\text{down}}$$
- **Stades Épars (2–4, `DeepSeekMoE`) :** Un expert partagé toujours actif $E_0$ + $E$ experts routés fins avec équilibrage **sans perte auxiliaire** par biais dynamique $b_i$ :
  $$s_{i,t} = \sigma(u_t^\top e_i), \qquad I_t = \text{TopK}_k\big(\{s_{i,t} + b_i\}_{i=1}^E\big)$$
  $$g_{i,t} = \frac{s_{i,t}}{\sum_{j \in I_t} s_{j,t} + \varepsilon} \quad (i \in I_t)$$
  $$\text{MoE}(u_t) = E_0(u_t) + \sum_{i \in I_t} g_{i,t} E_i(u_t)$$
  À la fin de chaque pas d'optimisation (hors graphe de gradient) :
  $$b_i \leftarrow b_i + \gamma_{\text{bal}} \cdot \text{sign}\!\left(\frac{k}{E} - f_i\right)$$
  où $f_i$ est la fraction empirique de tokens routés vers l'expert $i$. Zéro pollution du gradient principal !

### 1.5 Sortie Souveraine : Logit Softcap, Multi-Token Prediction & Mode Latent `Coconut`
- **Logit Softcap :** $z_t = C_{\text{cap}} \tanh\!\left(\frac{\text{RMSNorm}_{\text{final}}(h_t^{(L)}) W_{\text{head}}^\top}{C_{\text{cap}}}\right)$ avec $C_{\text{cap}} = 15.0$.
- **Multi-Token Prediction (MTP, $m=1$) :** Un bloc léger conditionné sur $[h_t^{(L)} \;\|\; E_{\text{tok}}[x_{t+1}]]$ prédit $x_{t+2}$ avec un poids de perte $\lambda_{\text{MTP}} = 0.3$ :
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(z_t, x_{t+1}) + \lambda_{\text{MTP}} \mathcal{L}_{\text{CE}}(z_t^{\text{MTP}}, x_{t+2})$$
- **Mode Pensée Latente Continue (`Coconut`) :** Entre les tokens `<|bot|>` et `<|eot|>`, le modèle court-circuite le décodage discret et réinjecte directement $\text{LayerNorm}(h_t^{(L)})$ comme vecteur d'embedding au pas $t+1$.

---

## 2. Théorèmes d'Invariance des 5 Opérateurs de Morphogenèse

Chaque bloc résiduel porte une porte scalaire apprenable $\alpha_\ell \in \mathbb{R}$ (initialisée à $1.0$ pour les blocs natifs) :
$$h^{(\ell)} = h^{(\ell-1)} + \alpha_\ell \, \mathcal{F}_\ell\big(\text{RMSNorm}_\ell(h^{(\ell-1)})\big)$$

### Théorème 1 — Croissance Interpositionnelle en Profondeur (`Gate-Zero`)
Soit un modèle à $L$ couches. Pour passer à $2L$ couches, chaque couche $\ell \in \{1, \dots, L\}$ engendre une couche jumelle $\ell'$ insérée immédiatement après $\ell$ avec :
$$\theta_{\ell'} \leftarrow \theta_\ell + \mathcal{N}(0, \sigma_{\text{init}}^2), \qquad \alpha_{\ell'} \leftarrow 0.0$$
**Preuve d'invariance :** À $t_0$, $\alpha_{\ell'} \mathcal{F}_{\ell'}(\cdot) = 0 \implies h_{\text{grown}}^{(2\ell)} \equiv h_{\text{old}}^{(\ell)}$ pour toute entrée $X$. De plus, $\frac{\partial \mathcal{L}}{\partial \alpha_{\ell'}} = \left\langle \frac{\partial \mathcal{L}}{\partial h^{(2\ell)}}, \mathcal{F}_{\ell'}(\text{RMSNorm}_{\ell'}(h^{(2\ell-1)})) \right\rangle \neq 0$, brisant immédiatement le plateau sans choc de gradient !

### Théorème 2 — Mitose Exacte des Neurones FFN (`Symmetric Split`)
Pour doubler la dimension interne d'un expert $\text{ReLU}^2$ de $p \to 2p$ ($W_{\text{up}} \in \mathbb{R}^{d \times p}$, $W_{\text{down}} \in \mathbb{R}^{p \times d}$) :
$$\widehat{W}_{\text{up}} = \big[W_{\text{up}} \;\; W_{\text{up}}\big] \in \mathbb{R}^{d \times 2p}, \qquad \widehat{W}_{\text{down}} = \frac{1}{2} \begin{bmatrix} W_{\text{down}} + \Delta \\ W_{\text{down}} - \Delta \end{bmatrix} \in \mathbb{R}^{2p \times d}$$
où $\Delta \sim \mathcal{N}(0, \sigma_{\text{split}}^2)$.
**Preuve d'invariance :** Soit $a = \text{ReLU}^2(u W_{\text{up}})$. Alors $\text{ReLU}^2(u \widehat{W}_{\text{up}}) = [a \;\; a]$, et :
$$[a \;\; a] \widehat{W}_{\text{down}} = \frac{1}{2} a (W_{\text{down}} + \Delta) + \frac{1}{2} a (W_{\text{down}} - \Delta) = a W_{\text{down}}$$
L'égalité fonctionnelle est **exacte** à $t_0$, tandis que $+\Delta$ et $-\Delta$ donnent immédiatement des gradients distincts aux deux neurones fils ($\frac{\partial \mathcal{L}}{\partial \widehat{W}_{\text{up}, 1}} \neq \frac{\partial \mathcal{L}}{\partial \widehat{W}_{\text{up}, 2}}$) !

### Théorème 3 — Élargissement Résiduel Compensé RMSNorm ($d \to \hat{d}$)
Soit $x \in \mathbb{R}^d$ étendu par zéros en $\hat{x} = [x \;\; \mathbf{0}_{\hat{d}-d}] \in \mathbb{R}^{\hat{d}}$.
Dans `RMSNorm`, $\text{RMS}(\hat{x}) = \sqrt{\frac{1}{\hat{d}} \sum_{i=1}^d x_i^2 + \varepsilon} = \sqrt{\frac{d}{\hat{d}}} \, \text{RMS}(x)$ (à $\varepsilon$ négligeable).
Donc, en définissant le nouveau vecteur de gain :
$$\hat{\gamma} = \left[ \gamma \sqrt{\frac{d}{\hat{d}}}, \;\; \mathbf{0}_{\hat{d}-d} \right]$$
on obtient exactement :
$$\frac{\hat{x}}{\text{RMS}(\hat{x})} \odot \hat{\gamma} = \left[ \frac{x}{\text{RMS}(x)} \odot \gamma, \;\; \mathbf{0}_{\hat{d}-d} \right]$$
En paddant de zéros les lignes/colonnes nouvelles des projections aval, le flux résiduel vit intégralement et identiquement dans le sous-espace $\mathbb{R}^d$ à $t_0$, puis colonise les $\hat{d}-d$ nouvelles dimensions dès les premiers pas de Muon !

### Théorème 4 — Upcycling Dense $\to$ MoE Sans Choc
Pour convertir un FFN dense $F(u)$ en couche `DeepSeekMoE` ($1_{\text{shared}} + E_{\text{routed}}$) :
$$E_0 \leftarrow \frac{1}{2} F \text{ (via } W_{\text{down}} \leftarrow \tfrac{1}{2} W_{\text{down}}\text{)}, \qquad E_i \leftarrow \frac{1}{2} F \text{ avec bruit antisymétrique sur } W_{\text{down}}$$
Comme $\sum_{i \in I_t} g_{i,t} = 1$, à $t_0$ (ou avec $\Delta = 0$ initial sur les experts routés + bruit sur le routeur), $E_0(u) + \sum_{i \in I_t} g_{i,t} E_i(u) = \frac{1}{2} F(u) + \frac{1}{2} F(u) = F(u)$ !

### Théorème 5 — Expansion Vocabulaire dans l'Enveloppe Convexe ($|\mathcal{V}_1| \to |\mathcal{V}_2|$)
Pour tout nouveau token BPE $v^* \in \mathcal{V}_2 \setminus \mathcal{V}_1$ décomposé par l'ancien tokenizer en $(u_1, \dots, u_m) \in \mathcal{V}_1^m$ :
$$E_{\text{tok}}^{\text{new}}[v^*] = \frac{1}{m} \sum_{j=1}^m E_{\text{tok}}^{\text{old}}[u_j], \qquad W_{\text{head}}^{\text{new}}[v^*] = \frac{1}{m} \sum_{j=1}^m W_{\text{head}}^{\text{old}}[u_j]$$
Garantit que les nouveaux tokens démarrent directement au barycentre sémantique exact de leurs morphèmes constituants.

---

## 3. Optimisation `CompositeMuon` + Fusion Perpétuelle `WSM`

### 3.1 Mise à jour `CompositeMuon`
Pour toute matrice cachée 2D $W \in \mathbb{R}^{m \times n}$ de gradient $G_t$ :
1. **Momentum de Nesterov :** $M_t = \mu M_{t-1} + G_t$, $\tilde{G}_t = G_t + \mu M_t$ ($\mu = 0.95$).
2. **Orthogonalisation de Newton-Schulz à 5 pas :** $O_t = \text{NS}_5(\tilde{G}_t / \|\tilde{G}_t\|_F)$ avec $(a, b, c) = (3.4445, -4.7750, 2.0315)$.
3. **Échelle RMS unifiée (`match_rms_adamw`) :** $\Delta W_t = 0.2 \sqrt{\max(m, n)} \, O_t$.
4. **Cautious Weight Decay :** Masque $C_t = \mathbb{I}(\Delta W_t \odot W_{t-1} > 0)$ :
   $$W_t = W_{t-1} - \eta \big(\Delta W_t + \lambda_{\text{wd}} (C_t \odot W_{t-1})\big)$$

### 3.2 Fusion Hors-Ligne `WSM (1-sqrt)`
Sur une fenêtre de $K$ checkpoints réguliers $\{\theta^{(1)}, \dots, \theta^{(K)}\}$ sauvegardés le long du tronc à LR constant $\eta_{\text{peak}}$ :
$$\tilde{w}_k = 1 - \sqrt{\frac{K - k}{K}}, \qquad w_k = \frac{\tilde{w}_k}{\sum_{j=1}^K \tilde{w}_j}, \qquad \theta_{\text{WSM}} = \sum_{k=1}^K w_k \theta^{(k)}$$
Les checkpoints les plus récents reçoivent le poids maximal selon une enveloppe concave duale au meilleur planning d'annealing `1-sqrt` (ICLR 2026).

---

## 4. Budget Mémoire & Calcul Multi-Stades

| Stade | $L$ | $d$ | $n_h$ | $d_{ff}$ / Experts | $|\mathcal{V}|$ | Params Totaux | Poids `fp32` / `bf16` | Empreinte Train RAM (Batch local) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stade 0 (`Seed-0`)** | 4 | 256 | 4 | 768 (Dense) | 8 192 | **~4.8M** | 19 Mo / 9.6 Mo | ~180 Mo (100 % fluide sur Xeon) |
| **Stade 1 (`Seed-1`)** | 8 | 384 | 6 | 1 152 (Dense) | 8 192 | **~18.5M** | 74 Mo / 37 Mo | ~420 Mo (tient dans RAM / L3) |
| **Stade 2 (`Sprout-MoE`)** | 12 | 512 | 8 | $1_S + 8_R \times 512$ (top-2) | 16 384 | **~68M** (26M act.) | 272 Mo / 136 Mo | ~1.1 Gio (Local + Burst GPU) |
| **Stade 3 (`Sapling-MoE`)** | 18 | 768 | 12 | $1_S + 16_R \times 768$ (top-2) | 32 768 | **~260M** (78M act.) | 1.04 Go / 520 Mo | Burst GPU Cloud + Inférence Locale |
| **Stade 4 (`Sovereign-MoE`)** | 24 | 1 024 | 16 | $1_S + 32_R \times 1024$ (top-4) | 65 536 | **~1.12B** (220M act.) | 4.48 Go / 2.24 Go | Burst Multi-GPU + Inférence Int8/BF16 |
