# NEVER-AI (`ChronoMorph-LM`)

**`never-ai`** est un projet d'ingénierie et de recherche visant à construire, entraîner de zéro et faire grandir continuellement un **Modèle de Langage Morphogénétique Souverain (`ChronoMorph-LM`)** couplé à un écosystème modulaire de **Plugins** et d'auto-apprentissage.

## Caractéristiques Architecturales Clés

- **Croissance Structurelle Sans Perte (*Zero-Shock Morphogenesis*) :** Passage progressif de `4.8M` $\to$ `18.5M` $\to$ `68M (MoE)` $\to$ `260M (MoE)` $\to$ `1.1B+ (MoE)` paramètres via des opérateurs préservant exactement la fonction ($f_{\theta_{\text{grown}}}(x) \equiv f_{\theta_{\text{old}}}(x)$) : clonage interpositionnel *Gate-Zero*, mitose symétrique FFN, élargissement résiduel compensé RMSNorm, *upcycling* `DeepSeekMoE` sans perte auxiliaire, et expansion vocabulaire dans l'enveloppe convexe.
- **Vitesse d'Apprentissage Maximale (`CompositeMuon` + `WSM`) :** Combinaison de l'optimiseur spectral **Muon** (Newton-Schulz à 5 pas sur les matrices cachées 2D) et d'**AdamW**, pilotée par l'ordonnancement perpétuel **Warmup-Stable-Merge (`WSM 1-sqrt`, ICLR 2026)** permettant un entraînement sans fin et une consolidation instantanée sans phase de decay destructive.
- **Plasticité Tri-Temporelle :**
  1. *Milliseconde (Inférence)* : **Neural Long-Term Memory (`Titans-MAG`)** mettant à jour un MLP mémoire en direct selon la surprise.
  2. *Minute/Heure (Sommeil)* : Auto-éditions synthétiques **`SEAL`** + réservoir de replay à 30 % + mémoire hybride auto-évolutive **`EvolveMem`**.
  3. *Jalon (Morphogenèse)* : Curriculum adaptatif multi-stades bilingue (FR/EN + Code + Math + Raisonnement).
- **Souveraineté Cognitive & Système Nerveux de Plugins :** Sanctuaire de pensée libre explicite (`<|think|>...<|/think|>`) et latent continu (**`Coconut`** `<|bot|>...<|eot|>`), zéro alignement de refus artificiel, routeur rapide **`ToolRAG`**, 7 plugins fondateurs et moteur **`SkillForge`** d'auto-création d'outils vérifiés.
- **Topologie Hybride Local + Cloud :** Entraînement continu vectorisé `AVX-512` (`fp32`) en local couplé à une synchronisation des checkpoints `safetensors` vers **Google Drive Vault** et worker **Burst GPU Cloud**.

## Documentation de Référence

1. **[`DIRECTIVES.md`](DIRECTIVES.md)** — **Plan Directeur & Les 12 Directives Globales de Travail** (Vision, Audit Matériel, 5 Piliers Scientifiques, Échelle des Stades 0→4, Feuille de Route en 7 Phases).
2. **[`docs/ARCHITECTURE_CHRONOMORPH.md`](docs/ARCHITECTURE_CHRONOMORPH.md)** — **Spécification Mathématique & Tensorielle** (MLA, Titans-MAG, DeepSeekMoE, Théorèmes d'Invariance Morphogénétique, Muon + WSM).
3. **[`docs/PLUGIN_AND_EVOLUTION_PROTOCOL.md`](docs/PLUGIN_AND_EVOLUTION_PROTOCOL.md)** — **Protocole des Plugins, `ToolRAG`, `EvolveMem`, `SEAL`, `GRPO` & Garde-Fous Anti-Goodhart**.
