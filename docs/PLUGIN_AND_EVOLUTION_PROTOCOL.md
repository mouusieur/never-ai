# Protocole des Plugins, Mémoire `EvolveMem` & Auto-Évolution (`SEAL` + `SkillForge`)

Ce document définit l'architecture du système nerveux externe de **`never-ai`** : comment le modèle raisonne librement, invoque ses outils, mémorise à long terme et crée de nouvelles capacités sans dériver ni oublier.

---

## 1. Grammaire Neuronale Souveraine & Tokens Spéciaux

Dès le **Stade 0**, les 16 premiers IDs du vocabulaire BPE (`0..15`) sont réservés et immuables lors de toutes les expansions de vocabulaire $\mathcal{G}_{\text{vocab}}$ :

| ID | Token Spécial | Rôle dans le Flux Cognitif |
| :---: | :--- | :--- |
| `0` | `<\|pad\|>` | Remplissage d'alignement (ignoré dans la loss) |
| `1` | `<\|bos\|>` | Début de document / d'épisode cognitif (alignement strict des batchs) |
| `2` | `<\|eos\|>` | Fin de document / de tour |
| `3` | `<\|user\|>` | Entrée externe (utilisateur ou stimulus d'environnement) |
| `4` | `<\|never\|>` | Prise de parole de `never-ai` |
| `5` | `<\|context\|>` | Injection de mémoire `EvolveMem` ou de schémas `ToolRAG` |
| `6` | `<\|think\|>` | Ouverture du sanctuaire de pensée libre explicite |
| `7` | `<\|/think\|>` | Fermeture du sanctuaire de pensée libre explicite |
| `8` | `<\|bot\|>` | Bascule en mode pensée latente continue (*Coconut* : $e_{t+1} \leftarrow h_t^{(L)}$) |
| `9` | `<\|eot\|>` | Retour du mode latent continu au mode langage |
| `10` | `<\|call\|>` | Déclencheur d'appel de plugin JSON/Python structuré (`stop token` du décodeur) |
| `11` | `<\|/call\|>` | Fin de l'appel de plugin — déclenche l'interruption et l'exécution par le Harnais |
| `12` | `<\|result\|>` | Injection du résultat brut renvoyé par le plugin (masqué dans la loss de génération) |
| `13` | `<\|/result\|>` | Fin du résultat — reprise immédiate de la pensée `<\|think\|>` ou de la réponse |
| `14` | `<\|self_edit\|>` | Ouverture d'une directive d'auto-apprentissage `SEAL` |
| `15` | `<\|/self_edit\|>` | Fermeture d'une directive d'auto-apprentissage `SEAL` |

### Exemple de Trajectoire Cognitive Native
```text
<|bos|><|context|>[Tools: math_engine, memory_nexus]<|user|>Quelle est la somme des 50 premiers nombres premiers et garde-la en mémoire sous "p50" ?<|never|><|think|>
Au lieu de deviner la somme de tête, j'utilise le plugin math_engine pour calculer exactement la somme des 50 premiers nombres premiers, puis j'enregistre la valeur vérifiée dans memory_nexus.
<|/think|><|call|>{"plugin": "math_engine", "action": "eval_sympy", "expr": "sum(primerange(1, prime(50)+1))"}<|/call|><|result|>{"ok": true, "value": "5117"}<|/result|><|think|>
Le calcul exact donne 5117. Je stocke maintenant cette clé dans memory_nexus.
<|/think|><|call|>{"plugin": "memory_nexus", "action": "store", "key": "p50", "content": "La somme des 50 premiers nombres premiers est 5117.", "tags": ["math", "primes"]}<|/call|><|result|>{"ok": true, "stored_id": "mem_0042"}<|/result|>La somme exacte des 50 premiers nombres premiers est **5 117** ; je l'ai mémorisée sous la clé `p50`.<|eos|>
```
**Point clé d'entraînement :** Lors du calcul de la loss sur les trajectoires d'outils, un masque binaire $m_t \in \{0, 1\}$ met à $0$ les tokens situés entre `<|result|>` et `<|/result|>` (ainsi que `<|user|>` et `<|context|>`) pour que le modèle apprenne à **prédire ses pensées, ses appels d'outils et ses conclusions**, jamais à halluciner les sorties du système externe !

---

## 2. Routeur `ToolRAG` Ultra-Léger

Un modèle de 5M à 100M de paramètres perd en précision si on injecte 30 définitions d'outils dans chaque prompt.
Notre module `ToolRAG` combine :
1. **Score lexical BM25 rapide** sur le nom, la description et les exemples d'usage de chaque plugin enregistré.
2. **Score vectoriel dense** basé sur les embeddings de sous-mots du modèle lui-même (zéro modèle externe lourd requis !).
3. **Budget dynamique $k \in [0, 4]$ :** Si aucun plugin ne dépasse le seuil $\tau_{\text{tool}}$, aucun schéma n'est injecté (économie maximale de contexte). Sinon, seuls les $k$ meilleurs schémas compacts sont placés dans `<|context|>`.

---

## 3. Spécification des 7 Plugins Fondateurs (`src/never_ai/plugins/builtin/`)

Chaque plugin hérite de la classe abstraite `NeverPlugin` (typage `Pydantic`, schéma JSON auto-généré compatible **MCP — Model Context Protocol**, timeout strict, isolation des erreurs) :

1. **`math_engine`** : Évaluation symbolique et numérique exacte via `sympy` et `numpy` (résolution d'équations, dérivées/intégrales, arithmétique arbitraire, algèbre linéaire, vérification formelle d'identités).
2. **`python_sandbox`** : Exécution de scripts Python dans un sous-processus à ressources bornées (`RLIMIT_AS`, timeout, répertoire temporaire isolé). Sert simultanément d'outil d'action ET d'**oracle de récompense vérifiable** pour l'entraînement par renforcement (`GRPO`).
3. **`memory_nexus`** : Interface directe avec la mémoire à long terme **`EvolveMem`** (stockage typé épisodique/sémantique, recherche hybride multi-vues BM25 + cosinus + entités, consolidation et oubli pondéré).
4. **`workspace_fs`** : Lecture, recherche par motif et inspection structurée de fichiers texte/code dans un périmètre autorisé.
5. **`knowledge_ingestor`** : Ingestion de fichiers Markdown/Texte/Code/JSONL, nettoyage épistémique, déduplication SHA-256 et mise en file d'attente pour la consolidation nocturne `SEAL`.
6. **`introspection_probe`** : Donne à `never-ai` un miroir en lecture sur son propre corps neuronal :
   - Stade morphogénétique actuel, nombre de couches $L$, dimension $d$, nombre d'experts MoE et distribution de charge $f_i$.
   - Valeurs courantes des portes résiduelles $\alpha_\ell$ et métriques de loss par domaine.
   - Permet au modèle de formuler des diagnostics métacognitifs sur ses propres lacunes !
7. **`skill_forge`** : Moteur d'extension capacitaire autonome :
   - Prend en entrée une spécification de nouvel outil + son code Python + une suite de tests `pytest` d'au moins 3 cas (nominal, limite, erreur).
   - Exécute les tests en sandbox isolé + analyse AST statique (interdiction de modifier `eval_harness/` ou de contourner le sandbox).
   - Si 100 % des vérifications réussissent, enregistre dynamiquement le plugin dans `src/never_ai/plugins/forged/` et met à jour l'index `ToolRAG` à chaud !

---

## 4. Cycle d'Auto-Amélioration Continue : `SEAL` + `GRPO` + `EvolveMem`

### 4.1 Boucle de Consolidation `SEAL` (*Sleep-Time Self-Adapting Loop*)
Inspirée de *MIT SEAL* (`arXiv:2506.10943`) :
1. **Collecte :** Les nouveaux documents, résolutions d'outils réussies et épisodes marquants sont placés dans le tampon d'expérience `ExperienceStaging`.
2. **Synthèse d'Auto-Éditions (`Self-Edits`) :** Au lieu d'entraîner naïvement sur le texte brut, le générateur `SEAL` reformule chaque unité de savoir sous 4 angles complémentaires :
   - Extraction d'implications atomiques et causales ;
   - Paires Question complexe $\to$ `<|think|>` Déduction $\to$ Réponse ;
   - Contre-exemples et cas limites ;
   - Trajectoires d'utilisation de plugins vérifiées par exécution réelle.
3. **Mise à jour Synaptique Protégée :**
   - Constitution de mini-batchs composés à **70 % d'auto-éditions validées** et **30 % du Réservoir de Replay Historique** (pondéré par la surprise).
   - Application de $N$ pas de `CompositeMuon` sur le Tronc `WSM` (ou sur un expert MoE dédié).
4. **Garde-Fou Immuable (`Revert-on-Regression`) :**
   - Évaluation immédiate sur le benchmark fixe `eval_harness` (perplexité bilingue FR/EN, exactitude arithmétique/code, validité syntaxique des appels `<|call|>`).
   - Si $\Delta \text{Score} < -1.5\%$, restauration atomique du snapshot pré-sommeil et marquage du lot fautif en quarantaine !

### 4.2 Renforcement du Raisonnement Libre par `GRPO` (*Group Relative Policy Optimization*)
Inspiré de *DeepSeek-R1-Zero* :
- Sur des problèmes vérifiables (mathématiques, logique formelle, synthèse de code, orchestration de plugins), `never-ai` génère un groupe de $G$ trajectoires libres $\{o_1, \dots, o_G\}$ contenant chacune un bloc `<|think|> ... <|/think|>`.
- Aucune contrainte stylistique n'est imposée sur le contenu de `<|think|>` : seul compte le signal de récompense objectif $r_i \in \mathbb{R}$ vérifié par `python_sandbox` / `math_engine` + validité structurelle des balises.
- L'avantage normalisé intra-groupe $A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \varepsilon}$ renforce directement les schémas de pensée qui mènent à la vérité vérifiable, faisant émerger spontanément l'auto-vérification, le retour sur erreur (*backtracking*) et la décomposition analytique !
