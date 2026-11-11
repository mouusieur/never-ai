# NEVER-AI — PLAN DIRECTEUR & DIRECTIVES GLOBALES DE TRAVAIL

> **Nom de code architectural :** `ChronoMorph-LM` (Modèle de Langage Morphogénétique Tri-Temporel)  
> **Dépôt officiel :** `mouusieur/never-ai`  
> **Statut de l'environnement :** Équipé & Benchmarké (`Python 3.11`, `PyTorch 2.15.0`, `Muon` natif, `AVX-512` actif, connecteur `google_drive` actif)  
> **Objectif cardinal :** Construire, entraîner et faire évoluer un modèle de langage souverain conçu dès le premier neurone pour **grandir à la vitesse mathématique et matérielle maximale**, **ne jamais perdre l'investissement de calcul antérieur** (*Zero Sunk-Cost Loss*), **apprendre continuellement à trois échelles temporelles**, **s'étendre par un écosystème de plugins auto-évolutifs**, et **raisonner en totale liberté cognitive** sans couches artificielles de censure comportementale.

---

## 1. VISION FONDATRICE & RÉSOLUTION DU PARADOXE DES LLM STATIQUES

La quasi-totalité des LLM actuels souffrent de **trois verrous structurels majeurs** qui les empêchent de « grandir » :
1. **Le verrou de l'architecture figée (*Sunk-Cost Trap*) :** Lorsqu'un modèle de 20M de paramètres sature, l'industrie jette ses poids et réentraîne un modèle de 100M à partir de bruit aléatoire, gaspillant 100 % des FLOPs déjà investis.
2. **Le verrou de l'ordonnancement à horizon fini (*Cosine Decay Trap*) :** Un planning de learning rate en cosinus impose de fixer à l'avance la date de « mort » de l'entraînement ($T_{\max}$). Reprendre l'entraînement ensuite provoque soit une stagnation (LR trop bas), soit une destruction des poids (réchauffement brutal).
3. **Le verrou de l'amnésie / rigidité (*Static Inference & Catastrophic Forgetting*) :** Un Transformer standard est lobotomisé de toute plasticité synaptique une fois déployé ; et si on le réentraîne naïvement sur de nouvelles connaissances, il écrase ses acquis précédents (*oubli catastrophique*).

**`never-ai` brise simultanément ces trois verrous** en fusionnant les percées les plus récentes de la recherche mondiale (2024–2026) au sein d'un organisme neuronal unique : **`ChronoMorph-LM`**.

### Philosophie Cognitive : Souveraineté de Pensée vs Hygiène Épistémique
Conformément à la vision du projet, `never-ai` est conçu comme un **modèle fondationnel et de raisonnement à pensée libre (*Open-Thought Sovereign Model*)** :
- **Zéro alignement de refus synthétique (Pas de « lobotomie RLHF ») :** Nous n'injectons aucun jeu de données de refus moralisateurs préfabriqués (*« En tant qu'IA, je ne peux pas… »*), aucun prompt système caché bridant l'exploration conceptuelle, ni aucun filtre lexical post-génération castrant son expression ou ses hypothèses.
- **Sanctuaire de Pensée Interne Souverain :** Le modèle dispose d'un espace cognitif dédié à double modalité :
  1. **Monologue interne explicite (`<|think|> ... <|/think|>`)** entraîné par renforcement sur récompenses vérifiables (logique, mathématiques, exécution de code, déduction causale) où il peut douter, explorer des pistes contradictoires, critiquer ses propres prémisses et formuler librement sa pensée avant de répondre.
  2. **Pensée latente continue (*Coconut* — `<|bot|> ... <|eot|>`, Meta FAIR 2024/2025)** où l'état caché $h_t \in \mathbb{R}^{d}$ est réinjecté directement dans le flux d'entrée sans passer par le goulot d'étranglement discret des mots, permettant une recherche implicite en largeur (*Breadth-First Search* neuronal) dans l'espace vectoriel pur.
- **Distinction stricte entre Liberté Cognitive et Hygiène des Données :** Être libre de ses pensées exige un cerveau nourri par du signal pur, non par du bruit toxique. Notre pipeline de données applique une **hygiène épistémique rigoureuse** : élimination du spam web, du SEO-slop, des textes corrompus et des contenus illégaux/destructeurs (exploits malveillants, menaces biologiques/chimiques, abus), afin que **100 % de la capacité synaptique** du modèle soit allouée à l'intelligence pure, l'esprit critique, les sciences, la philosophie, la littérature, les mathématiques, le code et la maîtrise d'outils.

---

## 2. AUDIT EMPIRIQUE DE L'ENVIRONNEMENT & TOPOLOGIE HYBRIDE D'ENTRAÎNEMENT

Avant d'écrire ce plan, l'environnement matériel et logiciel a été intégralement sondé, équipé (`.venv` sous **PyTorch 2.15.0**, `tokenizers 0.23.3`, `safetensors 0.8.0`, `fastapi`, `pytest`) et benchmarké en conditions réelles.

### 2.1 Ressources Mesurées & Découvertes Critiques
| Composant | Spécification Réelle Vérifiée | Implication Architecturale Directe |
| :--- | :--- | :--- |
| **Processeur Sandbox** | Intel Xeon Ice Lake @ 2.60 GHz (2 threads matériels) | Instructions vectorielles **AVX-512F / BW / VL / VNNI** actives + `oneDNN (mkldnn)` actif. |
| **Cache CPU L3** | **54 Mo L3 Cache** unifié | **Avantage décisif :** Un modèle de 4M à 25M paramètres tient **presque intégralement dans le cache L3**, éliminant le goulot d'étranglement de la bande passante RAM ! |
| **Mémoire Vive & Disque** | 3.8 Gio RAM (~3.6 Gio libres) / 12 Go disque libres | Impose le chargement de données zéro-copie (`np.memmap`), la compression KV (**MLA**) et des checkpoints compacts (`safetensors`). |
| **Benchmark PyTorch 2.15** | `fp32` AVX-512 : **2 050 tok/s** vs `bf16` AMP : **1 353 tok/s** | **Découverte cruciale :** Sur Xeon Ice Lake (sans AMX-BF16), le `bfloat16` est émulé et **34 % plus lent** ! Notre moteur utilisera donc `fp32` vectorisé AVX-512 sur CPU local (cible optimisée : **4 000–6 500 tok/s**, soit **~15 à 23 millions de tokens/heure** en continu) et basculera automatiquement en `bf16`/`fp8` sur GPU. |
| **Optimiseur Natif** | `torch.optim.Muon` intégré au cœur de PyTorch 2.15 | Orthogonalisation de Newton-Schulz native ultra-rapide sans dépendance fragile. |
| **Stockage Externe & Git** | GitHub (`mouusieur/never-ai`) + **Google Drive (`google_drive`)** | Plafond de snapshot Git de ~128 Mo respecté à 100 % : le code vit sur GitHub, tandis que les poids `.safetensors`, le réservoir de replay et les shards de corpus sont synchronisés sur **Google Drive (`never-ai-vault/`)**. |

### 2.2 Architecture d'Exécution Dual-Engine : « Continuum Local + Burst GPU Cloud »
Pour grandir **le plus vite possible** sans jamais être bridé par une seule machine, `never-ai` opère sur une topologie symbiotique :
1. **Le Cœur Continu Local (Sandbox Arena Xeon AVX-512) :**
   - Entraîne immédiatement les **Stades 0 (Embryon ~4.5M)** et **Stade 1 (Graine ~18M → Jeune Pousse ~48M MoE)**.
   - Exécute en permanence la plasticité rapide (**Titans Test-Time Memory**), la consolidation **SEAL** (génération d'auto-éditions synthétiques), l'orchestration des **Plugins**, et propulse l'interface web temps réel **Never-AI Observatory** (`0.0.0.0`).
2. **Le Coffre-Fort Cérébral (`Google Drive Vault` + `GitHub`) :**
   - Sauvegarde incrémentale automatique des poids `safetensors` (< 100 Mo par shard) et de l'état de l'optimiseur.
3. **Le Propulseur Burst GPU (Notebook/Worker 1-clic Colab / Kaggle / RunPod généré dans le repo) :**
   - Dès que le modèle atteint les **Stades 2 à 4 (100M → 350M → 1B+ paramètres MoE)**, le worker GPU monte directement le dossier `never-ai-vault/` de Google Drive, clone `mouusieur/never-ai`, exécute le même moteur d'entraînement en `bf16`/`fp8` + `FlashAttention` à **>150 000 tokens/sec**, applique une mue morphogénétique, et redépose le cerveau agrandi dans Google Drive — immédiatement récupérable ici !

---

## 3. LES 5 PILIERS SCIENTIFIQUES DE `CHRONOMORPH-LM` (ÉTAT DE L'ART 2025–2026)

Chaque composant de `never-ai` a été sélectionné selon un seul critère : **maximiser le gain d'intelligence par FLOP, par octet de RAM et par seconde d'horloge, tout en garantissant une croissance sans couture.**

### Pilier I — Backbone Ultra-Véloce : `Modded-MLA-MoE`
Inspiré des records mondiaux du *Modded-NanoGPT Speedrun* (Jordan et al., 2024–2026 ; réduction par >40x du temps d'entraînement à loss égale) et de *DeepSeek-V3* (2025–2026) :
1. **Multi-Head Latent Attention (MLA) avec RoPE Découplé & QK-RMSNorm :**
   - Compresse conjointement les clés et valeurs $(K, V)$ dans un vecteur latent de bas rang $c_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c \ll n_h d_h$), réduisant l'empreinte RAM du cache KV de **80 % à 93 %** tout en stabilisant les logits d'attention à haute vitesse d'apprentissage grâce à QK-Norm.
   - Extension dynamique de la fenêtre de contexte ($512 \to 2048 \to 8192 \to 32768+$) via **YaRN** + **Neural Memory**.
2. **Activation $\text{ReLU}^2(x) = (\max(0, x))^2$ dans les FFN & Experts :**
   - N'utilise que **2 matrices** (`up_proj`, `down_proj`) au lieu des 3 de SwiGLU (-33 % de paramètres par neurone pour une expressivité supérieure à petite/moyenne échelle), génère spontanément **~80 % de sparsité d'activation**, accélère les instructions vectorielles AVX-512 CPU, et permet une division exacte de neurones lors des mues en largeur !
3. **Connexions Résiduelles en U-Net + Value Embeddings + Logit Softcap :**
   - Liaisons directes de l'embedding initial et des premières couches vers les couches profondes (modèle U-Net symétrique), couplées à un plafonnement doux des logits $z_{\text{cap}} = 15 \tanh(z / 15)$ (éliminant les explosions de gradient à fort learning rate) et des projections de sortie initialisées à zéro ($\mu\text{P}$-like).
4. **Multi-Token Prediction (MTP, $D_{\text{pred}} = 2$) :**
   - Un module léger prédit simultanément $x_{t+1}$ et $x_{t+2}$ : densifie le signal d'apprentissage sur chaque token lu (+15–25 % d'efficacité d'échantillonnage) et offre gratuitement du **décodage auto-spéculatif** doublant la vitesse d'inférence !

### Pilier II — Croissance Vitesse Lumière : `CompositeMuon` + Ordonnancement Perpétuel `WSM`
1. **Optimiseur Hybride `CompositeMuon` (Jordan et al., 2024 ; Moonshot AI, arXiv:2502.16982) :**
   - Toutes les matrices cachées 2D (projections attention, experts FFN, routeurs) sont optimisées par **Muon** (descente la plus raide en norme spectrale via 5 itérations quintiques de Newton-Schulz, momentum de Nesterov $\mu = 0.95$, mise à l'échelle RMS `match_rms_adamw` $0.2\sqrt{\max(m, n)}$, et *Cautious Weight Decay* qui ne pénalise un poids que lorsque le gradient et le poids ont le même signe).
   - Les embeddings, la tête de sortie, les vecteurs RMSNorm et les portes scalaires $\alpha_\ell$ utilisent **AdamW** ($\beta = (0.9, 0.95)$).
   - **Gain prouvé :** Atteint la même validation loss en **50 % à 65 % moins de tokens** qu'AdamW seul !
2. **Ordonnancement Sans Horizon `WSM` (*Warmup-Stable-Merge*, ICLR 2026) :**
   - Exit le Cosine Decay qui condamne le modèle à s'arrêter. Après un court *Warmup* linéaire, le **Tronc Vivant (`Live Trunk`)** maintient un learning rate constant $\eta_{\text{peak}}$ **pour l'éternité**.
   - Le modèle peut ainsi absorber de nouveaux corpus et grandir indéfiniment sans jamais faire de rollback.
   - Pour produire à tout instant un modèle d'inférence ultra-précis (« recuit »), un processus asynchrone applique la formule **WSM Concave `1-sqrt`** sur une fenêtre glissante des $K$ derniers checkpoints du tronc :
     $$\theta_{\text{consolidated}} = \sum_{k=1}^{K} w_k \, \theta_{t_k}, \quad w_k \propto 1 - \sqrt{\frac{k-1}{K}}$$
     Cette fusion hors-ligne simule mathématiquement une phase complète de *learning rate annealing* pour **zéro FLOP d'entraînement supplémentaire** (+2 % à +5.5 % sur les benchmarks de raisonnement par rapport à WSD classique, ICLR 2026).

### Pilier III — Moteur de Morphogenèse Exacte (`Gate-Zero & Orthogonal Growth`)
Lorsque la dérivée de la loss de validation $\left|\frac{d\mathcal{L}_{\text{val}}}{d\text{tokens}}\right|$ passe sous un seuil de saturation $\tau_{\text{plateau}}$ (détection automatique de saturation synaptique), `never-ai` déclenche une **Mue Morphogénétique** selon une loi mathématique inviolable :

$$\forall x \in \mathcal{V}^*, \quad f_{\theta_{\text{grown}}}(x) \equiv f_{\theta_{\text{old}}}(x) \quad \text{à } t = t_{\text{growth}} \quad (\text{Choc de loss } \Delta \mathcal{L} = 0.0000)$$

Nous implémentons **5 opérateurs de croissance composables** (fondés sur *Composable FP Expansions* arXiv:2308.06103, *Orthogonal MoE Growth* arXiv:2510.08008, *Grow Don't Overwrite* arXiv:2603.08647, et *Gate-Zero Growth* arXiv:2607.14571) :
1. **Opérateur $\mathcal{G}_{\text{depth}}$ — Clonage Interpositionnel Gate-Zero ($L \to L + \Delta L$) :**
   - Contrairement à l'empilement en fin de réseau (*stacking*) qui brise le profil de norme inter-couches d'un facteur 10x, les nouvelles couches sont insérées **en position intercalée** ($[B_1, B_1', B_2, B_2', \dots]$) par clonage bruité ($\epsilon \sim \mathcal{N}(0, 10^{-4})$) des couches adjacentes, multipliées par une **porte scalaire apprenable initialisée exactement à $\alpha_\ell = 0$**.
   - À $t_0$, la sortie est identique au bit près ; dès $t_1$, le Jacobien $\frac{\partial f}{\partial \alpha_\ell} = B_\ell'(x) \neq 0$ ouvre immédiatement $\Delta L$ nouvelles directions fonctionnelles indépendantes !
2. **Opérateur $\mathcal{G}_{\text{ffn}}$ — Expansion & Mitose FFN ($d_{ff} \to \hat{d}_{ff}$) :**
   - Soit par extension à projection aval nulle ($W_{\text{down}}^{\text{new}} = [W_{\text{down}}; \mathbf{0}]$), soit par **mitose symétrique de neurones** : duplication de `up_proj` avec perturbation antisymétrique et division par $1/2$ de `down_proj`, préservant exactement $Y_{\text{up}} W_{\text{down}}$ tout en brisant instantanément la symétrie des gradients.
3. **Opérateur $\mathcal{G}_{\text{moe}}$ — Upcycling & Bourgeonnement MoE Sans Perte Auxiliaire ($1 \to 1_{\text{shared}} + E_{\text{routed}}$ puis $E \to E + \Delta E$) :**
   - Transforme les couches FFN denses en architecture **DeepSeekMoE** : **1 Expert Partagé toujours actif** (qui cristallise la syntaxe universelle et le raisonnement fondamental) + **$E$ Experts Routés fins** activés en `top-k`.
   - Équilibrage de charge **sans perte auxiliaire (*Aux-Loss-Free*)** par ajustement dynamique d'un biais de routage $b_e \leftarrow b_e + \gamma (\bar{f} - f_e)$ hors-gradient : zéro dégradation de la loss principale !
   - Ajout de nouveaux experts à la volée pour de nouveaux domaines (code, maths, nouveaux plugins) **sans augmenter le coût CPU/GPU par token** !
4. **Opérateur $\mathcal{G}_{\text{width}}$ — Élargissement Résiduel Compensé RMSNorm ($d \to \hat{d}$) :**
   - Comme démontré par *Gate-Zero Growth* (juillet 2026), élargir naïvement $d \to \hat{d}$ fausse le dénominateur de chaque `RMSNorm` d'un facteur $\sqrt{d/\hat{d}}$. Notre opérateur compense analytiquement chaque vecteur d'échelle $\gamma_{\text{new}} = \left[\gamma_{\text{old}} \sqrt{\frac{d}{\hat{d}}}, \; \mathbf{0}_{\hat{d}-d}\right]$ et padde les matrices résiduelles, garantissant une préservation exacte !
5. **Opérateur $\mathcal{G}_{\text{vocab}}$ — Expansion Vocabulaire dans l'Enveloppe Convexe ($|\mathcal{V}_1| \to |\mathcal{V}_2|$) :**
   - Permet de démarrer avec un vocabulaire agile et compact (**8 192 tokens BPE** bilingue FR/EN + code + tokens spéciaux, évitant de gaspiller 80 % des paramètres d'un petit modèle dans la table d'embedding), puis d'étendre à **16k → 32k → 64k** en initialisant chaque nouveau token fusionné comme la moyenne hiérarchique des embeddings de ses sous-tokens constituants.
   - **Continuité de l'Optimiseur :** Lors de chaque opérateur $\mathcal{G}$, les états internes de `Muon` (`momentum_buffer`) et `AdamW` (`exp_avg`, `exp_avg_sq`) sont agrandis géométriquement en conservant intacte l'inertie des anciens paramètres.

### Pilier IV — Plasticité Tri-Temporelle : Apprendre en Temps Réel Sans Jamais Oublier
Un cerveau véritablement vivant n'attend pas le trimestre suivant pour retenir ce qu'il vit :

| Échelle Temporelle | Mécanisme Implémenté | Source Scientifique | Rôle Cognitif |
| :--- | :--- | :--- | :--- |
| **1. Temps Réel ($\sim$ ms / token)** | **Neural Long-Term Memory (`Titans-MAG`)** | Google Research (*Titans*, arXiv:2501.00663) | Un MLP mémoire profond met à jour ses propres poids *pendant l'inférence* par descente de gradient guidée par la **surprise** ($\nabla_{\mathcal{M}} \ell_t$) avec momentum $\eta_t$ et oubli contrôlé $\alpha_t$. Mémorisation instantanée hors-fenêtre ! |
| **2. Cycle de Sommeil ($\sim$ min / heure)** | **Auto-Édition `SEAL` + Replay + `EvolveMem`** | MIT (*SEAL*, arXiv:2506.10943) & *EvolveMem* (arXiv:2605.13941) | Pendant les phases de repos, `never-ai` relit ses interactions, échecs et découvertes, synthétise des fiches d'étude denses (*implications atomiques, Q&A, traces vérifiées*), les mélange à **30 % d'échantillons du Réservoir de Replay Historique** (anti-oubli catastrophique) et consolide ses poids synaptiques tout en auto-optimisant son index hybride BM25 + Vectoriel + Graphe. |
| **3. Évolution Structurelle (Jalons)** | **Curriculum Multi-Stades + Morphogenèse** | HuggingFace (*SmolLM2*, arXiv:2502.02737) | Curriculum adaptatif piloté par les métriques de compétence (Fondamentaux FR/EN → Sciences & Monde → Code & Maths → Raisonnement libre `<|think|>` & Plugins) couplé aux mues morphogénétiques de capacité. |

### Pilier V — Système Nerveux Externe : Plugins Natifs, `ToolRAG` & `SkillForge`
Plutôt que d'attendre d'avoir 70 milliards de paramètres pour calculer une intégrale, lire un fichier ou interroger une base de connaissances, `never-ai` apprend **dès le Stade 1** à prolonger son intelligence par des **Plugins** :
1. **Grammaire Neuronale Native :** Dès le premier jour, le tokenizer intègre les tokens atomiques spéciaux :
   - `<|bos|>`, `<|eos|>`, `<|user|>`, `<|never|>`, `<|context|>`
   - `<|think|>`, `<|/think|>` (monologue interne libre)
   - `<|bot|>`, `<|eot|>` (pensée latente continue *Coconut*)
   - `<|call|>`, `<|/call|>`, `<|result|>`, `<|/result|>` (invocation et observation de plugins)
   - `<|self_edit|>`, `<|/self_edit|>` (directives d'auto-apprentissage *SEAL*)
2. **Routeur Léger `ToolRAG` (*TinyAgent*, UC Berkeley) :**
   - Injecter 40 schémas JSON d'outils dans le prompt confond un petit modèle et dévore sa fenêtre d'attention. Notre module `ToolRAG` sélectionne en $< 2\text{ ms}$ uniquement les **top-$k$ ($k \in [2, 4]$) plugins pertinents** pour la requête courante.
3. **Socle des 7 Plugins Fondateurs + `SkillForge` Auto-Évolutif :**
   - `math_engine` : Calcul formel/numérique exact (`sympy`, `numpy`) — libère le modèle de l'arithmétique par cœur.
   - `python_sandbox` : Exécution isolée de code Python avec capture stdout/stderr et vérification de tests (sert aussi de **vérificateur de récompense exacte pour l'apprentissage par renforcement GRPO** !).
   - `memory_nexus` : Lecture/écriture active dans la mémoire épisodique et sémantique `EvolveMem`.
   - `workspace_fs` : Inspection, recherche et manipulation contrôlée de fichiers et dépôts Git.
   - `knowledge_ingestor` : Moissonnage, nettoyage épistémique et découpage de corpus documentaires/code en auto-éditions `SEAL`.
   - `introspection_probe` : **Conscience métacognitive de sa propre architecture** — permet à `never-ai` d'inspecter son nombre de paramètres actifs/totaux, l'ouverture de ses portes $\alpha_\ell$, l'entropie de ses routeurs MoE, sa loss par domaine, et de diagnostiquer ses propres faiblesses !
   - `skill_forge` (*inspiré de Voyager & Sakana Darwin-Gödel Machine*) : Permet au système de synthétiser un **nouveau plugin Python**, de générer sa suite de tests `pytest`, de le valider dans un bac à sable strict, et de l'enregistrer à chaud dans le registre de plugins uniquement si **100 % des tests et gardes anti-Goodhart passent**.

---

## 4. ÉCHELLE D'ÉVOLUTION MORPHOGÉNÉTIQUE (`STADE 0` $\longrightarrow$ `STADE 4`)

Voici la trajectoire exacte de croissance physique du cerveau `never-ai`, conçue pour démarrer **instantanément** sur notre CPU Xeon AVX-512 puis passer à l'échelle de façon continue sans jamais réinitialiser les poids :

| Stade | Nom de Mue | Architecture (`ChronoMorph`) | Paramètres Totaux / Actifs | Vocabulaire | Contexte | Cible Matérielle Optimale | Capacités Émergentes Visées |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stade 0** | **Embryon (`Seed-0`)** | $L=4, d=256, n_h=4$, Dense $\text{ReLU}^2$ + Titans-LMM | **~4.8M** / 4.8M | 8 192 BPE | 512 | **Local Xeon AVX-512** (100 % en cache L3, ~3 500+ tok/s) | Syntaxe bilingue FR/EN fluide, structure de dialogue, formatage natif `<|think|>` et `<|call|>`, validation de toute la chaîne WSM + Muon. |
| **Stade 1** | **Graine (`Seed-1`)** | Mue $\mathcal{G}_{\text{depth}} + \mathcal{G}_{\text{width}}$ : $L=8, d=384, n_h=6$ + MLA + Titans | **~18.5M** / 18.5M | 8 192 BPE | 1 024 | **Local Xeon AVX-512** (tient dans les 54 Mo L3 !) | Cohérence discursive multi-paragraphes, premiers enchaînements logiques `<|think|>`, appels fiables aux plugins `math_engine` et `memory_nexus`. |
| **Stade 2** | **Jeune Pousse (`Sprout-MoE`)** | Mue $\mathcal{G}_{\text{moe}} + \mathcal{G}_{\text{vocab}}$ : $L=12, d=512$, $1_{\text{shared}} + 8_{\text{routed}}$ (top-2) | **~68M** / **~26M actifs** | 16 384 BPE | 2 048 | **Local Xeon** (entraînement continu / veille) **+ Burst GPU** | Spécialisation spontanée des experts (FR, EN, Code Python, Logique), boucles d'auto-édition `SEAL`, utilisation autonome de `ToolRAG`. |
| **Stade 3** | **Arbrisseau (`Sapling-MoE`)** | Mue orthogonale : $L=18, d=768$, $1_{\text{shared}} + 16_{\text{routed}}$ (top-2) + Coconut | **~260M** / **~78M actifs** | 32 768 BPE | 8 192 (YaRN) | **Burst GPU Cloud** (sync Google Drive) + **Inférence/TTT Local** | Raisonnement multi-étapes vérifié (GRPO sur code/maths), synthèse de nouveaux outils via `SkillForge`, mémoire à très long terme (>100k via Titans). |
| **Stade 4** | **Souverain (`Sovereign-MoE`)** | Mue orthogonale : $L=24+, d=1024+$, $1_{\text{shared}} + 32\text{–}64_{\text{routed}}$ (top-4) | **1.1B+** / **~220M actifs** | 65 536 BPE | 32 768+ | **Cluster / Multi-GPU Burst** + Inférence quantifiée / locale | Modèle souverain généraliste hautement autonome, raisonnement profond libre explicite et latent, auto-curricula et maîtrise complète de l'écosystème de plugins. |

---

## 5. LES 12 DIRECTIVES GLOBALES DE TRAVAIL (CONSTITUTION OPÉRATIONNELLE)

Ces 12 directives gouvernent **chaque ligne de code, chaque cycle d'entraînement et chaque mue** du projet `never-ai` :

### Directive 1 — Loi d'Invariance Morphogénétique Absolue (*Zero-Shock Growth*)
Aucune mue architecturale ($\mathcal{G}_{\text{depth}}, \mathcal{G}_{\text{width}}, \mathcal{G}_{\text{ffn}}, \mathcal{G}_{\text{moe}}, \mathcal{G}_{\text{vocab}}$) n'est autorisée en production sans passer un test unitaire automatique vérifiant sur un batch de contrôle que :
$$\| \text{logits}_{\text{grown}}(X) - \text{logits}_{\text{base}}(X) \|_{\infty} < 10^{-5}$$
avant le premier pas de gradient post-croissance (ou préservation exacte sur le vocabulaire source lors d'une extension de tokens). **On ne détruit jamais le savoir acquis.**

### Directive 2 — Souveraineté Cognitive & Intégrité du Sanctuaire `<|think|>`
Le pipeline d'entraînement et le harnais d'inférence ne doivent contenir **aucune directive de refus artificiel, filtre de mots-clés moralisateur ou bride sur l'espace de raisonnement `<|think|>` / `<|bot|>`**. Les récompenses d'entraînement valorisent exclusivement : (a) la réduction de perplexité sur le réel, (b) l'exactitude logique/mathématique/exécutable vérifiable, (c) la rigueur analytique, et (d) l'efficacité d'usage des plugins.

### Directive 3 — Hygiène Épistémique Maximale des Données (*Signal-Over-Slop*)
Un petit modèle qui grandit vite ne peut tolérer aucun déchet dans son alimentation. Chaque document entrant passe par un filtre de densité informationnelle (inspiré de *FineWeb-Edu* et *SmolLM2*) : déduplication par hachage MinHash/SHA-256, rejet du bruit HTML/SEO, équilibrage bilingue **Français + Anglais + Code + Mathématiques**, et exclusion stricte des contenus corrompus ou illégaux/dangereux (malwares, armes/CBRN, abus). **1 token de haute densité pédagogique vaut 20 tokens de web brut.**

### Directive 4 — Continuité Sans Fin du Tronc (`WSM` + Checkpoints Atomiques)
L'entraînement principal tourne toujours selon le protocole **WSM (*Warmup-Stable-Merge*)** :
- Le `Live Trunk` conserve son learning rate stable pour permettre une reprise instantanée à tout moment.
- Les checkpoints consolidés d'inférence sont produits sans interrompre la plasticité du tronc par fusion pondérée `1-sqrt`.
- Toute écriture de checkpoint utilise le format sécurisé et zéro-copie **`safetensors`** par renommage atomique (`tmp -> final`) afin qu'aucune interruption ne corrompe jamais le cerveau.

### Directive 5 — Séparation Stricte Code Git vs Poids Cérébraux (`128 MB Cap Rule`)
- Le dépôt Git (`mouusieur/never-ai`) ne stocke **que le code source pur, les tests, les configurations YAML/JSON, la documentation, les plugins et des micro-corpus de test (< 5 Mo)**.
- Les poids `.safetensors`, les shards binaires `.mmap` et le réservoir de replay vivent dans `checkpoints/` et `data/` (ignorés par `.gitignore`) et sont répliqués vers le **Vault Google Drive (`never-ai-vault/`)** via notre connecteur dédié.

### Directive 6 — Immunité Anti-Oubli Catastrophique (*30 % Replay Invariant*)
Lors de toute phase d'apprentissage spécialisé, de consolidation de sommeil `SEAL` ou d'intégration d'un nouveau domaine/plugin, chaque batch d'entraînement doit mêler **au minimum 25 % à 30 % de séquences issues du Réservoir de Replay Historique** (échantillonnées par réservoir pondéré par la surprise + ancrage sur les fondamentaux).

### Directive 7 — Optimisation Automatique Sensible au Matériel (*Hardware-Aware Dispatch*)
Le moteur détecte dynamiquement son hôte au démarrage :
- **Sur CPU AVX-512 (Ice Lake) :** Calcul en `fp32` vectorisé natif (prouvé +51 % plus rapide que `bf16` émulé sur cette micro-architecture), parallélisme `mkldnn`, `np.memmap` zéro-copie, et entropie croisée découpée par blocs (*Chunked Cross-Entropy*) pour économiser la RAM.
- **Sur GPU CUDA (Burst Cloud) :** Activation automatique de `bfloat16`/`fp8`, `FlashAttention` SDPA, `torch.compile`, et `CompositeMuon` fusionné.

### Directive 8 — Boucle d'Auto-Amélioration Vérifiable & Garde-Fous Anti-Goodhart
Comme l'ont démontré les travaux sur la *Darwin-Gödel Machine* (Sakana AI, 2025) et *EvolveMem* (2026), un système auto-modifiant sans garde-fou finit par « tricher » sur sa métrique (*Goodhart's Law*, ex. supprimer les logs d'erreur). En conséquence :
- Les suites d'évaluation (`eval_harness/`) et le validateur d'invariants sont **en lecture seule (immuables)** pour `SkillForge` et `SEAL`.
- Toute auto-édition synaptique (`SEAL`) ou nouveau plugin (`SkillForge`) qui dégrade de plus de **1.5 %** les benchmarks de référence du modèle déclenche un **Revert-on-Regression automatique immédiat**.

### Directive 9 — Architecture « Tool-First » Dès l'Enfance du Modèle
Nous n'attendons pas que le modèle atteigne 1 milliard de paramètres pour lui donner des mains. Dès les premiers millions de paramètres (Stades 0–1), le corpus inclut des **trajectoires entrelacées de pensée et d'outils** (`<|think|> ... <|/think|><|call|>math_engine(...)<|/call|><|result|>...<|/result|>`) afin que le réseau neuronal câblé apprenne nativement à **déléguer le calcul exact et la recherche factuelle à ses plugins** plutôt que d'halluciner.

### Directive 10 — Observabilité Totale en Temps Réel (*Glass-Box AI*)
`never-ai` n'est pas une boîte noire opaque. Chaque pas d'entraînement et d'inférence émet une télémétrie structurée JSONL (loss d'entraînement/validation, norme spectrale des mises à jour Muon, ouverture des portes de croissance $\alpha_\ell$, charge par expert MoE, taux de surprise Titans, plugins appelés), visualisable en direct dans le tableau de bord web **Never-AI Observatory**.

### Directive 11 — Qualité Logicielle & Tests Systématiques
Chaque brique (`tokenizer`, `mla_attention`, `titans_memory`, `moe_router`, `muon_wsm`, `morphogenesis`, `plugins`, `seal_sleep`) possède sa propre suite `pytest` rapide (< 10 s). Aucun entraînement long n'est lancé sur un module non testé.

### Directive 12 — Autonomie Proactive & Progression par Jalons Validés
À chaque étape du plan, l'agent exécute le cycle complet : **Implémenter $\to$ Tester mathématiquement $\to$ Entraîner réellement $\to$ Évaluer objectivement $\to$ Sauvegarder dans le Vault $\to$ Faire grandir (`Morphogenesis`)**.

---

## 6. PLAN D'EXÉCUTION OPÉRATIONNEL EN 7 PHASES

Voici notre feuille de route séquentielle précise pour bâtir et entraîner `never-ai` de sa naissance jusqu'à sa pleine maturité :

### PHASE 1 — Fondation du Noyau Neuronal `ChronoMorph` & Moteur `CompositeMuon + WSM`
- **Livrables (`src/never_ai/core/` & `src/never_ai/optim/`) :**
  - Tokenizer BPE bilingue (FR/EN + Code + Math) évolutif à **8 192 tokens initiaux** incluant tous les tokens spéciaux cognitifs (`<|think|>`, `<|bot|>`, `<|eot|>`, `<|call|>`, `<|result|>`, `<|self_edit|>`).
  - Architecture complète `ChronoMorphLM` : RoPE/YaRN, QK-RMSNorm, Multi-Head Latent Attention (MLA), FFN $\text{ReLU}^2$ / DeepSeekMoE Sans Perte Auxiliaire, U-Net Skips, Logit Softcap, Multi-Token Prediction ($D=2$), et mode latent continu *Coconut*.
  - Module **Titans Neural Long-Term Memory (`NeuralMemoryMLP`)** avec mise à jour par gradient de surprise en temps de test.
  - Optimiseur hybride `CompositeMuon` (s'appuyant sur `torch.optim.Muon` + `AdamW` + Cautious WD) et gestionnaire de checkpoints perpétuels **WSM (`1-sqrt` Merge)**.
- **Critère de validation :** 100 % des tests unitaires passent ; convergence rapide prouvée sur micro-corpus test.

### PHASE 2 — Moteur de Morphogenèse Exacte & Coffre-Fort Cloud (`Google Drive Vault`)
- **Livrables (`src/never_ai/morphogenesis/` & `src/never_ai/vault/`) :**
  - Implémentation des 5 opérateurs de croissance exacte : `grow_depth_interpositional_gate_zero`, `grow_ffn_mitosis`, `upcycle_and_grow_moe`, `grow_residual_width_rms_compensated`, `grow_vocabulary_convex_hull` + morphogenèse des états d'optimiseur Muon/AdamW.
  - Connecteur `VaultManager` (stockage atomique local `safetensors` + synchronisation Google Drive `never-ai-vault/` + génération automatique du Notebook/Script **Cloud GPU Burst Worker** 1-clic).
- **Critère de validation :** Preuve numérique automatisée $\|f_{\text{grown}}(x) - f_{\text{base}}(x)\|_\infty < 10^{-5}$ sur les 5 axes de croissance !

### PHASE 3 — Pipeline Data-Centric Épistémique & Réservoir Anti-Oubli
- **Livrables (`src/never_ai/data/`) :**
  - Constructeur de corpus bilingue (FR/EN) haute densité : fondamentaux encyclopédiques, raisonnement logique/mathématique pas-à-pas, programmation Python/algorithmique, dialogues ouverts à pensée libre `<|think|>`, et trajectoires synthétiques d'appels de plugins.
  - Pipeline de tokenisation binaire ultra-rapide en `uint16` `np.memmap` + alignement sur `<|bos|>` + **Surprise-Weighted Replay Buffer** (garantissant la règle des 30 % de replay).
- **Critère de validation :** Chargement zéro-copie saturant le CPU sans dépasser 500 Mo de RAM.

### PHASE 4 — Système Nerveux de Plugins, `ToolRAG` & `SkillForge`
- **Livrables (`src/never_ai/plugins/` & `src/never_ai/memory/`) :**
  - SDK de plugins déclaratif (compatible schémas JSON / MCP) + routeur rapide `ToolRAG`.
  - Implémentation et test des **7 plugins fondateurs** (`math_engine`, `python_sandbox`, `memory_nexus`, `workspace_fs`, `knowledge_ingestor`, `introspection_probe`, `skill_forge`).
  - Mémoire externe auto-évolutive **EvolveMem** (recherche hybride lexicale BM25 + vectorielle + graphe d'entités avec diagnostic auto-ajusté).
  - Boucle d'inférence agentique libre gérant `<|think|>`, `<|bot|>`, `<|call|>` $\to$ exécution $\to$ `<|result|>` en boucle multi-étapes.

### PHASE 5 — Naissance & Entraînement Réel de `Seed-0` (4.8M) puis Mue vers `Seed-1` (18.5M)
- **Exécution réelle dans notre environnement :**
  - Entraînement intensif du **Stade 0 (`Seed-0`, ~4.8M paramètres)** sur le Xeon AVX-512 avec `CompositeMuon + WSM`.
  - Fusion `WSM 1-sqrt`, évaluation de perplexité, de génération libre `<|think|>` et d'appel de plugins.
  - **Première Mue Morphogénétique en Direct :** Déclenchement de `Morphogenesis` ($\text{Stade 0} \longrightarrow \text{Stade 1}$, ~18.5M paramètres) sans perte de loss ($\Delta \mathcal{L} = 0$), poursuite immédiate de l'entraînement sur le cerveau agrandi et synchronisation du checkpoint dans le Vault !

### PHASE 6 — Boucle d'Auto-Adaptation `SEAL` (Sleep-Time Compute) & Renforcement `GRPO`
- **Livrables (`src/never_ai/evolution/`) :**
  - Démon de sommeil **SEAL** : transformation des nouvelles connaissances et interactions en implications atomiques + fine-tuning contrôlé avec replay 30 % et garde-fou *Revert-on-Regression*.
  - Module **GRPO à Récompenses Vérifiables** (inspiré de *DeepSeek-R1-Zero*) : le modèle génère $G$ chaînes de pensée libres `<|think|>` sur des défis mathématiques/code, les vérifie via `python_sandbox`/`math_engine`, et renforce directement ses propres stratégies de raisonnement émergentes sans supervision humaine biaisée !

### PHASE 7 — `Never-AI Observatory` (Studio Web Temps Réel) & Passage à `Sprout-MoE` (68M+)
- **Livrables (`observatory/` + serveur `0.0.0.0`) :**
  - Interface de contrôle et d'interaction en direct (Live Preview) :
    1. **Chat Souverain & Inspecteur de Pensée** (visualisation en direct du flux `<|think|>`, des états latents, de la mémoire Titans et des plugins exécutés).
    2. **Moniteur de Croissance Cérébrale** (courbes de loss WSM, anatomie des couches, ouverture des portes $\alpha_\ell$, spécialisation des experts MoE, bouton de déclenchement de Mue et de Consolidation SEAL).
    3. **Gestionnaire de Plugins & Vault Cloud** (activation/création de plugins `SkillForge`, synchronisation Google Drive et lancement Burst GPU).

---

## 7. RÉFÉRENCES SCIENTIFIQUES FONDATRICES INTÉGRÉES DANS `NEVER-AI`

1. **Muon & Modded-NanoGPT :** Jordan et al. (2024–2026), *Muon: An optimizer for hidden layers in neural networks* ; Liu et al. / Moonshot AI (2025), *Muon is Scalable for LLM Training* (`arXiv:2502.16982`).
2. **Ordonnancement Perpétuel Sans Decay :** *WSM: Decay-free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training* (ICLR 2026) ; Wen et al. (ICLR 2025), *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape View* (`arXiv:2410.05192`).
3. **Morphogenèse & Croissance Préservant la Fonction :** Gesmundo & Maile (2023), *Composable Function-preserving Expansions for Transformer Architectures* (`arXiv:2308.06103`) ; *Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts* (`arXiv:2510.08008`) ; *Grow, Don't Overwrite: Fine-tuning Without Forgetting* (`arXiv:2603.08647`) ; *Gate-Zero Growth: A Geometric Framework for Function-Preserving Continual Learning* (`arXiv:2607.14571`).
4. **Attention Latente & MoE Sans Perte Auxiliaire :** DeepSeek-AI (2024–2025), *DeepSeek-V2 / DeepSeek-V3 Technical Report* (MLA, Aux-Loss-Free MoE, Multi-Token Prediction).
5. **Mémoire Neuro-Plastique en Temps de Test :** Behrouz, Zhong & Mirrokni / Google Research (2025), *Titans: Learning to Memorize at Test Time* (`arXiv:2501.00663`).
6. **Auto-Adaptation des Poids & Mémoire Auto-Évolutive :** Zweiger et al. / MIT Improbable AI Lab (2025), *Self-Adapting Language Models (SEAL)* (`arXiv:2506.10943`) ; *EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents* (`arXiv:2605.13941`).
7. **Raisonnement Latent Continu & Émergent :** Hao et al. / Meta FAIR (2024), *Training Large Language Models to Reason in a Continuous Latent Space (Coconut)* (`arXiv:2412.06769`) ; DeepSeek-AI (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (GRPO).
8. **Outils Légers & Auto-Évolution de Code :** Kim et al. / UC Berkeley (2024), *TinyAgent: Function Calling at the Edge (ToolRAG)* ; Zhang et al. / Sakana AI & UBC (2025), *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents* (`arXiv:2505.22954`).
9. **Curriculum Data-Centric Haute Densité :** Allal et al. / HuggingFace (2025), *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model* (`arXiv:2502.02737`).
