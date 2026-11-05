# NEVER-AI — Direction Artistique & Identité Visuelle (« Monolith Noir »)

## 1. Philosophie Visuelle : L'Esthétique du Silence & de la Lame

Inspirée directement de la radicalité graphique de la **Landing Page et du parcours Auth de `x.com`**, la direction artistique de **Never-AI** refuse les dégradés néon artificiels, les violets « IA générique » et le bruit visuel.

L'interface repose sur trois lois plastiques :
1. **Le Vide Noir OLED (`#000000`)** : La surface primaire n'est pas gris foncé mais **noir absolu `#000000`**. L'intelligence émerge du néant par pur contraste typographique.
2. **La Géométrie Chirurgicale (`1px #2F3336`)** : Aucune ombre portée floue dans l'espace de travail ; la structure spatiale est découpée au scalpel par des filets capillaires `1px solid #2F3336` et des en-têtes vitrés translucides (`rgba(0, 0, 0, 0.65)` + `backdrop-filter: blur(12px)`).
3. **La Dualité Blanc Pur / Acier** : Les actions décisives sont des pilules pleines (`border-radius: 9999px`) en blanc franc `#EFF3F4` (texte `#0F1419`), tandis que les actions secondaires sont cerclées d'acier `#536471`.

---

## 2. Le Monogramme Officiel `NEVER` (Double-Lame Morphogénétique)

À l'image du `X` géométrique à double trait (un fût plein croisant un fût évidé en réserve), l'emblème de **Never-AI** est un **`N` / Pont Synaptique Monumental** inscrit dans un carré parfait `512 × 512` :
- **Deux piliers verticaux biseautés** représentent la mémoire immuable consolidée (`WSM`) et le cortex actif (`ChronoMorph-LM`).
- **Une diagonale creuse à double rail parallèle** relie le sommet gauche à la base droite : elle incarne le **canal de pensée continue (`<|think|>` / `Coconut`)** et la **croissance morphogénétique sans choc** où l'information circule sans jamais être réinitialisée.
- **Usage** :
  - **Monumental (`320px – 380px`)** dans la moitié gauche de la Landing Page `#000000`.
  - ** Miniature (`28px – 32px`)** au sommet centré des modales d'authentification et en haut du rail de navigation gauche.
  - **Icône Exécutable Windows (`assets/brand/never_ai.ico`)** : glyphe blanc pur sur squircle noir `#000000` bordé d'un liseré acier `#2F3336`, multi-résolution (`16px` $\to$ `256px`).

---

## 3. Tokens de Design Officiels (`Monolith Noir`)

| Token CSS | Valeur | Rôle dans l'interface |
| :--- | :--- | :--- |
| `--bg-void` | `#000000` | Fond principal Landing, Modales et Colonne de discussion |
| `--bg-elevated` | `#16181C` | Cartes latérales (Anatomie, SEAL), blocs `<|think|>`, bulles secondaires |
| `--bg-hover` | `#181818` / `rgba(231,233,234,0.08)` | Survol des éléments de navigation et fils de discussion |
| `--modal-backdrop` | `rgba(91, 112, 131, 0.40)` | Voile ardoise caractéristique des modales d'auth `x.com` |
| `--border-hairline` | `#2F3336` | Filets de séparation `1px` entre les 3 colonnes |
| `--border-input` | `#333639` | Bordure au repos des champs à *floating label* |
| `--text-primary` | `#E7E9EA` | Texte principal, titres massifs et réponses de Never-AI |
| `--text-secondary` | `#71767B` | Métadonnées, horodatage, compteurs de paramètres, labels au repos |
| `--cta-white` | `#EFF3F4` (hover `#D7DBDC`) | Boutons Pill primaires & bouton d'envoi |
| `--accent-focus` | `#1D9BF0` | Bordure active des *floating labels*, indicateur d'entraînement live |

---

## 4. Parcours Utilisateur de l'Application `.exe`

1. **Landing Split-Screen (`x.com` Signature)** :
   - Colonne gauche : Monogramme `NEVER` géant sur `#000000`.
   - Colonne droite : Accroche massive **`La pensée libre. Sans fin.`** + sous-titre **`Discutez avec Never-AI sur votre machine.`** + pile de boutons Pill (`Créer un profil souverain`, `Accès rapide invité`, séparateur `ou`, `Se connecter`).
2. **Modale Auth Locale (`rgba(91, 112, 131, 0.4)` Backdrop)** :
   - Fenêtre noire `#000000` (`border-radius: 16px`), fermeture `✕` en haut à gauche, monogramme centré, champs à **labels flottants animés** (`@identifiant`, nom d'affichage, clé locale optionnelle hachée en SHA-256 — zéro cloud, zéro télémétrie externe).
3. **Workspace Desktop 3 Colonnes** :
   - **Gauche** : Navigation X-style (`Explorer / Discuter`, `Anatomie & Mue`, `Plugins & Forge`, `Consolidation SEAL`), historique des conversations multi-tours, badge utilisateur `@handle`.
   - **Centre** : Fil de conversation épuré avec header vitré `blur(12px)`, inspection pliable du sanctuaire `<|think|>`, exécution d'outils vérifiés et compositeur minimaliste.
   - **Droite** : Télémétrie cérébrale temps réel en cartes `#16181C` (`Sprout-MoE`, courbe de loss monochrome blanc, barres `Gate-Zero`, déclenchement de mue morphogénétique).
