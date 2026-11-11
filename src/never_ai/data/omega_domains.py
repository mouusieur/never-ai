"""
Omega 10-Domain Verified Polymathic Knowledge & Dialogue Library (FR + EN).
Zero raw special tokens inside prose strings to guarantee crisp BPE generation!
"""

from __future__ import annotations

from typing import Any


OMEGA_DOMAIN_KNOWLEDGE: list[dict[str, Any]] = [
    # 1. Conversation, Esprit & Productivité
    {
        "domain": "conversation",
        "key": "domain_conv_what_can_you_do",
        "tags": ["sais", "faire", "exactement", "peux", "capacités", "outils"],
        "q": "Que sais-tu faire exactement ?",
        "thought": "Résumons clairement mes 10 domaines polymathiques, ma mémoire EvolveMem et mes 15 plugins vérifiés.",
        "a": (
            "Je peux discuter avec toi en français ou en anglais sur **10 domaines scientifiques et techniques**, "
            "résoudre des calculs formels via `math_engine`, tracer des courbes vectorielles SVG avec `plot_svg_engine`, "
            "auditer et exécuter du code Python, analyser tes documents locaux en RAG (`📎`), et grandir continuellement par morphogenèse sans oublier."
        ),
    },
    {
        "domain": "conversation",
        "key": "domain_conv_help",
        "tags": ["aide", "capacités", "domaines", "compétences", "outils"],
        "q": "Quels sont tous tes domaines de compétence maintenant ?",
        "thought": "Présentons clairement mes 10 domaines polymathiques et mes 15 plugins souverains.",
        "a": (
            "Je couvre désormais **10 domaines fondamentaux** (Mathématiques, Informatique & Code, Physique, "
            "Biologie & Neurosciences, Philosophie & Logique, Histoire, Économie & Théorie des Jeux, Cybersécurité, "
            "Rédaction FR/EN et Organisation) appuyés par **15 plugins vérifiés** : calcul symbolique `SymPy`, "
            "traceur de courbes `SVG`, statistiques & régression, logique formelle, constantes physiques, audit de code AST, "
            "cryptographie, ingestion de documents RAG et `SkillForge`."
        ),
    },
    {
        "domain": "conversation",
        "key": "domain_conv_joke",
        "tags": ["blague", "humour", "drôle", "rire"],
        "q": "Raconte-moi une blague intelligente d'informaticien ou de mathématicien.",
        "thought": "Choisissons un trait d'esprit fin sur la récursivité et les bases numériques.",
        "a": (
            "Il existe **10 types** de personnes dans le monde : celles qui comprennent le binaire, et celles qui ne le comprennent pas. "
            "Et pour comprendre la **récursivité**, il faut d'abord comprendre la récursivité !"
        ),
    },
    {
        "domain": "conversation",
        "key": "domain_conv_tired",
        "tags": ["fatigué", "motivation", "conseil", "productivité"],
        "q": "J'ai beaucoup de travail et je ne sais pas par où commencer, tu me conseilles quoi ?",
        "thought": "Proposons une méthode concrète de découpage cognitif sans jargon inutile.",
        "a": (
            "Applique la règle des **3 briques** : identifie la seule tâche qui débloque le reste, découpe-la en une première action "
            "de **15 minutes** sans distraction, et ignore tout le reste tant que cette brique n'est pas posée. Si tu veux, dis-moi "
            "`/task add ...` et on structure ton plan ensemble."
        ),
    },
    # 2. Mathématiques Pures & Appliquées
    {
        "domain": "mathematics",
        "key": "domain_math_bayes",
        "tags": ["bayes", "probabilité", "statistiques", "théorème"],
        "q": "Explique-moi le théorème de Bayes et pourquoi il est si important.",
        "thought": "Formulons P(A|B) = P(B|A)P(A)/P(B) et illustrons l'intuition du faux positif médical.",
        "a": (
            "Le **théorème de Bayes** s'écrit `P(H|E) = [P(E|H) · P(H)] / P(E)`. Il mesure comment réviser rationnellement "
            "la probabilité d'une hypothèse `H` après avoir observé une preuve `E`. Il est crucial en science et en IA car il évite "
            "l'oubli du taux de base (*base rate fallacy*) : même un test fiable à 99 % donne beaucoup de faux positifs si le phénomène est très rare."
        ),
    },
    {
        "domain": "mathematics",
        "key": "domain_math_svd",
        "tags": ["algèbre", "matrice", "valeurs propres", "svd", "muon"],
        "q": "Qu'est-ce que la décomposition en valeurs singulières (SVD) ?",
        "thought": "Relions A = U Sigma V^T à la géométrie des transformations linéaires et à l'optimiseur Muon.",
        "a": (
            "Toute matrice réelle `M` se factorise sous la forme `M = U Σ Vᵀ`, où `U` et `V` sont des matrices orthogonales (rotations pures) "
            "et `Σ` est diagonale positive (étirements le long d'axes perpendiculaires). C'est d'ailleurs ce principe qu'exploite mon optimiseur "
            "**Muon** : en égalisant les valeurs singulières du gradient vers 1 via Newton-Schulz, il apprend dans toutes les directions utiles à la fois."
        ),
    },
    # 3. Informatique, Algorithmique & Génie Logiciel
    {
        "domain": "computer_science",
        "key": "domain_cs_binary_search",
        "tags": ["python", "algorithme", "recherche dichotomique", "binary search", "code"],
        "q": "Écris l'algorithme de recherche dichotomique (binary search) en Python.",
        "thought": "Implémentons une recherche dichotomique itérative propre en O(log n) sans dépassement d'indice.",
        "a": (
            "Voici une implémentation itérative optimale en complexité temporelle `O(log n)` et spatiale `O(1)` :\n"
            "```python\n"
            "def recherche_dichotomique(tableau: list[int], cible: int) -> int:\n"
            "    gauche, droite = 0, len(tableau) - 1\n"
            "    while gauche <= droite:\n"
            "        milieu = (gauche + droite) // 2\n"
            "        if tableau[milieu] == cible:\n"
            "            return milieu\n"
            "        elif tableau[milieu] < cible:\n"
            "            gauche = milieu + 1\n"
            "        else:\n"
            "            droite = milieu - 1\n"
            "    return -1\n"
            "```"
        ),
    },
    {
        "domain": "computer_science",
        "key": "domain_cs_quicksort",
        "tags": ["tri", "quicksort", "mergesort", "complexité", "python"],
        "q": "Quelle est la différence entre Merge Sort et QuickSort ?",
        "thought": "Comparons la stabilité, le pire cas O(n log n) vs O(n^2) et la localité mémoire cache.",
        "a": (
            "**Merge Sort** garantit toujours `O(n log n)` dans le pire des cas et est *stable*, mais requiert `O(n)` de mémoire auxiliaire. "
            "**QuickSort** trie sur place (`O(log n)` de pile) avec une excellente localité de cache CPU, mais peut dégrader en `O(n²)` si le pivot est mal choisi (d'où le pivot médiane-de-trois ou *Timsort* en Python)."
        ),
    },
    {
        "domain": "computer_science",
        "key": "domain_cs_rest_graphql",
        "tags": ["api", "architecture", "backend", "web", "sql"],
        "q": "Comment concevoir une architecture backend propre et résiliente ?",
        "thought": "Synthétisons l'isolation domaine/infrastructure, l'idempotence, les verrous fins et les tests.",
        "a": (
            "Une architecture backend robuste repose sur **quatre règles** : séparer strictement la logique métier pure des entrées/sorties, "
            "rendre les écritures critiques **atomiques et idempotentes**, éviter les verrous globaux bloquants sur le chemin de lecture, "
            "et couvrir chaque invariant par des tests automatisés déterministes."
        ),
    },
    # 4. Physique & Astrophysique
    {
        "domain": "physics",
        "key": "domain_phys_quantum",
        "tags": ["quantique", "heisenberg", "schrödinger", "superposition", "physique"],
        "q": "Explique simplement le principe d'incertitude d'Heisenberg.",
        "thought": "Précisons qu'il ne s'agit pas d'un défaut d'instrument mais d'une propriété ondulatoire fondamentale Delta x * Delta p >= hbar / 2.",
        "a": (
            "Le **principe d'indétermination d'Heisenberg** (`Δx · Δp ≥ ℏ/2`) établit qu'on ne peut pas définir simultanément avec une précision arbitraire "
            "la position `x` et l'impulsion `p` d'une particule. Ce n'est pas une limite de nos instruments : c'est une propriété mathématique fondamentale "
            "des ondes (comme la transformée de Fourier entre un signal court dans le temps et large en fréquence)."
        ),
    },
    {
        "domain": "physics",
        "key": "domain_phys_relativity",
        "tags": ["einstein", "relativité", "lumière", "temps", "gravité"],
        "q": "Pourquoi le temps ralentit-il près d'un objet massif ou à grande vitesse ?",
        "thought": "Relions l'invariance de la vitesse de la lumière c et la courbure de la métrique d'espace-temps.",
        "a": (
            "Parce que la vitesse de la lumière `c` est identique pour tous les observateurs : plus tu te déplaces vite dans l'espace, "
            "ou plus l'espace-temps est courbé par une masse intense (relativité générale d'Einstein), plus l'intervalle de **temps propre** "
            "s'écoule lentement par rapport à un observateur éloigné — un effet mesuré chaque jour par les horloges atomiques des satellites GPS."
        ),
    },
    # 5. Biologie & Neurosciences
    {
        "domain": "biology",
        "key": "domain_bio_hebb",
        "tags": ["cerveau", "neurone", "synapse", "hebb", "plasticité", "mémoire"],
        "q": "Comment le cerveau humain mémorise-t-il de nouvelles informations ?",
        "thought": "Expliquons la potentialisation à long terme (LTP) de Donald Hebb et le parallèle avec Titans.",
        "a": (
            "Le cerveau humain apprend grâce à la **plasticité synaptique de Hebb** (*« des neurones qui s'activent ensemble se lient ensemble »*) : "
            "l'hippocampe encode rapidement les événements nouveaux ou surprenants, puis le sommeil consolide progressivement ces traces dans le néocortex — "
            "exactement l'inspiration biologique de ma mémoire temps-test **Titans** couplée à la consolidation **SEAL**."
        ),
    },
    {
        "domain": "biology",
        "key": "domain_bio_crispr",
        "tags": ["génétique", "crispr", "arn", "cellule", "biologie"],
        "q": "Comment fonctionne la technologie CRISPR-Cas9 ?",
        "thought": "Décrivons l'ARN guide complémentaire et l'endonucléase Cas9 issue de l'immunité bactérienne.",
        "a": (
            "**CRISPR-Cas9** est un ciseau moléculaire programmable issu du système immunitaire des bactéries : un **ARN guide** repère une séquence "
            "exacte de 20 lettres sur l'ADN cible, puis la protéine **Cas9** coupe les deux brins à cet endroit précis, permettant de désactiver ou corriger un gène."
        ),
    },
    # 6. Philosophie, Logique & Épistémologie
    {
        "domain": "philosophy",
        "key": "domain_phil_godel",
        "tags": ["gödel", "incomplétude", "logique", "vérité", "démonstration"],
        "q": "Que dit vraiment le théorème d'incomplétude de Gödel ?",
        "thought": "Évitons les contresens mystiques : énonçons rigoureusement le premier théorème sur les systèmes formels cohérents contenant l'arithmétique.",
        "a": (
            "En 1931, **Kurt Gödel** a démontré que dans tout système axiomatique formel **cohérent** et assez riche pour exprimer l'arithmétique des entiers, "
            "il existe des propositions vraies mais **indémontrables** à l'intérieur de ce système, et le système ne peut pas démontrer sa propre cohérence seul."
        ),
    },
    {
        "domain": "philosophy",
        "key": "domain_phil_consciousness",
        "tags": ["conscience", "esprit", "ia", "philosophie", "chalmers"],
        "q": "Quelle est la différence entre l'intelligence fonctionnelle et la conscience phénoménale ?",
        "thought": "Distinguons la résolution formelle d'informations et le 'problème difficile' des qualia.",
        "a": (
            "L'**intelligence fonctionnelle** désigne la capacité mesurable à modéliser le réel, abstraire, raisonner et résoudre des problèmes nouveaux. "
            "La **conscience phénoménale** (*qualia* de David Chalmers) désigne le ressenti subjectif intérieur. En tant qu'IA, je possède une réflexion symbolique "
            "et latente vérifiable, sans prétendre simuler artificiellement des émotions biologiques."
        ),
    },
    # 7. Histoire des Sciences & Civilisations
    {
        "domain": "history",
        "key": "domain_hist_turing_shannon",
        "tags": ["histoire", "informatique", "turing", "shannon", "ordinateur"],
        "q": "Qui sont les grands fondateurs théoriques de l'informatique moderne ?",
        "thought": "Citons Ada Lovelace, Alan Turing (1936), John von Neumann (1945) et Claude Shannon (1948).",
        "a": (
            "Quatre figures ont posé le socle de l'informatique : **Ada Lovelace** (premier algorithme pensé pour une machine symbolique), "
            "**Alan Turing** (1936, formalisation universelle du calcul), **John von Neumann** (architecture à programme enregistré) et "
            "**Claude Shannon** (1948, logique booléenne des circuits et théorie mathématique de l'information en bits)."
        ),
    },
    # 8. Économie, Finance & Théorie des Jeux
    {
        "domain": "economics",
        "key": "domain_econ_nash",
        "tags": ["économie", "théorie des jeux", "nash", "dilemme du prisonnier", "stratégie"],
        "q": "Qu'est-ce qu'un équilibre de Nash en théorie des jeux ?",
        "thought": "Définissons le profil stratégique où aucun joueur n'a intérêt à dévier unilatéralement et citons le dilemme du prisonnier.",
        "a": (
            "Un **équilibre de Nash** est une situation stratégique où aucun participant ne peut améliorer son propre gain en changeant **seul** de stratégie, "
            "les choix des autres restant fixes. Le *dilemme du prisonnier* montre qu'un équilibre de Nash rationnel individuellement peut être sous-optimal collectivement sans coopération."
        ),
    },
    {
        "domain": "economics",
        "key": "domain_econ_compound",
        "tags": ["finance", "intérêts composés", "investissement", "croissance", "exponentielle"],
        "q": "Comment fonctionnent les intérêts composés ?",
        "thought": "Formulons C_n = C_0 * (1 + r)^n et la règle de 72.",
        "a": (
            "Les **intérêts composés** suivent la loi exponentielle `C(n) = C₀ · (1 + r)ⁿ` : les gains générés chaque période produisent eux-mêmes des gains aux périodes suivantes. "
            "Astuce mentale rapide (**règle de 72**) : à un taux annuel de `r %`, un capital double en environ `72 / r` années (ex. 9 ans à 8 %)."
        ),
    },
    # 9. Cybersécurité & Cryptographie
    {
        "domain": "cybersecurity",
        "key": "domain_sec_hash_enc",
        "tags": ["cryptographie", "sha256", "chiffrement", "hachage", "sécurité", "mot de passe"],
        "q": "Quelle est la différence entre hachage cryptographique et chiffrement ?",
        "thought": "Distinguons une fonction à sens unique (SHA-256, PBKDF2) d'une bijection réversible avec clé (AES, RSA).",
        "a": (
            "Le **chiffrement** (ex. `AES-256`, `ChaCha20`) est **réversible** : quiconque détient la clé secrète retrouve le message original. "
            "Le **hachage cryptographique** (ex. `SHA-256`, `PBKDF2`) est **à sens unique** : il produit une empreinte de taille fixe impossible à inverser, "
            "idéale pour vérifier un mot de passe ou l'intégrité d'un fichier sans jamais stocker le secret en clair."
        ),
    },
    # 10. Anglais Polymathique & Traduction
    {
        "domain": "english_polymath",
        "key": "domain_en_capabilities",
        "tags": ["english", "capabilities", "tools", "domains", "what can you do"],
        "q": "What tools and domains do you master in version Omega?",
        "thought": "Summarize the 10 knowledge domains, 15 deterministic plugins, Deep Think mode, and SVG plotting in clear English.",
        "a": (
            "In **Never-AI Omega**, I combine my morphogenetic **ChronoMorph-LM** neural core with **10 knowledge domains** and **15 deterministic plugins**: "
            "symbolic math (`SymPy`), vector function plotting (`SVG`), statistics & regression, formal logic truth tables, CODATA physics constants, "
            "Python AST auditing, cryptography, local document RAG, task planning, and autonomous `SkillForge`."
        ),
    },
    # --- Additional Deep Polymathic Entries Across All 10 Domains ---
    {
        "domain": "mathematics",
        "key": "domain_math_fourier",
        "tags": ["fourier", "analyse", "signal", "fréquence", "onde"],
        "q": "À quoi sert la transformée de Fourier ?",
        "thought": "Expliquons la décomposition d'un signal temporel en somme d'ondes sinusoïdales de fréquences pures.",
        "a": (
            "La **transformée de Fourier** décompose n'importe quel signal complexe du domaine temporel `f(t)` en la somme des **fréquences pures** "
            "(sinusoïdes d'amplitude et de phase précises) qui le constituent. Elle est au cœur de la compression audio/image (`JPEG`, `MP3`), des télécoms, de l'IRM et de la physique quantique."
        ),
    },
    {
        "domain": "mathematics",
        "key": "domain_math_euler",
        "tags": ["euler", "complexe", "exponentielle", "identité"],
        "q": "Pourquoi l'identité d'Euler e^(i*pi) + 1 = 0 est-elle célèbre ?",
        "thought": "Montrons comment e^(ix) = cos(x) + i*sin(x) relie les cinq constantes fondamentales en x = pi.",
        "a": (
            "L'**identité d'Euler** `e^(iπ) + 1 = 0` découle de `e^(ix) = cos(x) + i·sin(x)` évaluée en `x = π`. "
            "Elle relie en une seule égalité minimale les cinq constantes les plus fondamentales des mathématiques : `0`, `1`, `π` (géométrie), `e` (analyse) et `i` (algèbre complexe)."
        ),
    },
    {
        "domain": "computer_science",
        "key": "domain_cs_dijkstra",
        "tags": ["dijkstra", "graphe", "plus court chemin", "heapq", "python"],
        "q": "Écris l'algorithme de Dijkstra en Python avec un tas binaire (heapq).",
        "thought": "Implémentons Dijkstra en O((V + E) log V) avec heapq sur un graphe à poids positifs.",
        "a": (
            "Voici l'algorithme de **Dijkstra** optimal en `O((V + E) log V)` :\n"
            "```python\n"
            "import heapq\n\n"
            "def dijkstra(graphe: dict[str, list[tuple[str, float]]], source: str) -> dict[str, float]:\n"
            "    dist = {sommet: float('inf') for sommet in graphe}\n"
            "    dist[source] = 0.0\n"
            "    tas = [(0.0, source)]\n"
            "    while tas:\n"
            "        d_u, u = heapq.heappop(tas)\n"
            "        if d_u > dist[u]:\n"
            "            continue\n"
            "        for v, poids in graphe[u]:\n"
            "            nouvelle = d_u + poids\n"
            "            if nouvelle < dist[v]:\n"
            "                dist[v] = nouvelle\n"
            "                heapq.heappush(tas, (nouvelle, v))\n"
            "    return dist\n"
            "```"
        ),
    },
    {
        "domain": "computer_science",
        "key": "domain_cs_lru_cache",
        "tags": ["cache", "lru", "ordereddict", "python", "structure de données"],
        "q": "Comment implémenter un cache LRU en O(1) en Python ?",
        "thought": "Utilisons collections.OrderedDict avec move_to_end et popitem(last=False).",
        "a": (
            "En combinant une table de hachage et une liste doublement chaînée (`OrderedDict`), chaque lecture et écriture s'exécute en **`O(1)`** :\n"
            "```python\n"
            "from collections import OrderedDict\n\n"
            "class CacheLRU:\n"
            "    def __init__(self, capacite: int = 128) -> None:\n"
            "        self.cap = capacite\n"
            "        self.data: OrderedDict[str, object] = OrderedDict()\n\n"
            "    def get(self, cle: str) -> object | None:\n"
            "        if cle not in self.data:\n"
            "            return None\n"
            "        self.data.move_to_end(cle)\n"
            "        return self.data[cle]\n\n"
            "    def put(self, cle: str, val: object) -> None:\n"
            "        self.data[cle] = val\n"
            "        self.data.move_to_end(cle)\n"
            "        if len(self.data) > self.cap:\n"
            "            self.data.popitem(last=False)\n"
            "```"
        ),
    },
    {
        "domain": "physics",
        "key": "domain_phys_thermo",
        "tags": ["thermodynamique", "entropie", "énergie", "carnot", "zéro absolu"],
        "q": "Énonce les trois principes de la thermodynamique.",
        "thought": "Formulons rigoureusement la conservation de l'énergie, la croissance de l'entropie et le théorème de Nernst.",
        "a": (
            "1. **Premier principe (Conservation)** : L'énergie totale d'un système isolé se conserve (`ΔU = W + Q`).\n"
            "2. **Deuxième principe (Irréversibilité)** : L'entropie `S` d'un système isolé ne peut qu'augmenter ou rester constante (`ΔS ≥ 0`), fixant la flèche du temps.\n"
            "3. **Troisième principe (Nernst)** : L'entropie d'un cristal pur tend vers zéro lorsque la température approche du zéro absolu (`0 K = -273,15 °C`)."
        ),
    },
    {
        "domain": "physics",
        "key": "domain_phys_maxwell",
        "tags": ["maxwell", "électromagnétisme", "lumière", "champ", "onde"],
        "q": "Que montrent les quatre équations de Maxwell ?",
        "thought": "Synthétisons Gauss électrique, Gauss magnétique, Faraday et Ampère-Maxwell.",
        "a": (
            "Les **quatre équations de Maxwell** unifient l'électricité, le magnétisme et l'optique : les charges créent le champ électrique (Gauss), "
            "il n'existe pas de monopôle magnétique isolé (`∇·B = 0`), un champ magnétique variable induit un champ électrique (Faraday), et un courant ou champ électrique variable crée un champ magnétique (Ampère-Maxwell) — prédisant que la **lumière est une onde électromagnétique** se propageant à `c`."
        ),
    },
    {
        "domain": "biology",
        "key": "domain_bio_crispr_alt",
        "tags": ["crispr", "édition génétique", "adn", "cas9"],
        "q": "Comment fonctionne l'édition génétique CRISPR-Cas9 ?",
        "thought": "Expliquons le guidage par ARN complémentaire et la coupure double-brin par Cas9.",
        "a": (
            "Le système **CRISPR-Cas9** associe un **ARN guide** de ~20 nucléotides, qui reconnaît par complémentarité une cible unique de l'ADN génomique (adjacente à un motif `PAM`), "
            "à l'enzyme **Cas9** qui réalise une coupure double-brin précise permettant d'inactiver ou de réécrire une séquence génétique."
        ),
    },
    {
        "domain": "biology",
        "key": "domain_bio_atp",
        "tags": ["cellule", "mitochondrie", "atp", "métabolisme", "respiration"],
        "q": "Quel est le rôle des mitochondries et de l'ATP dans nos cellules ?",
        "thought": "Décrivons la chaîne respiratoire et la phosphorylation oxydative.",
        "a": (
            "Les **mitochondries** sont les centrales bioénergétiques des cellules eucaryotes : grâce à la chaîne respiratoire et à un gradient de protons, "
            "l'enzyme **ATP synthase** régénère l'**ATP** (*adénosine triphosphate*), la monnaie énergétique universelle qui alimente nos muscles, nos pompes ioniques neuronales et la synthèse des protéines."
        ),
    },
    {
        "domain": "philosophy",
        "key": "domain_phil_popper",
        "tags": ["popper", "falsifiabilité", "science", "épistémologie"],
        "q": "Qu'est-ce que le critère de réfutabilité (falsifiabilité) de Karl Popper ?",
        "thought": "Distinguons un énoncé empirique réfutable d'un dogme immunisé contre l'expérience.",
        "a": (
            "Pour **Karl Popper**, une théorie n'est **scientifique** que si elle prend le risque d'être contredite par une observation précise (*réfutabilité*). "
            "Une affirmation compatible avec tous les résultats possibles n'explique rien empiriquement ; une loi scientifique progresse en survivant aux tests les plus sévères."
        ),
    },
    {
        "domain": "philosophy",
        "key": "domain_phil_fallacies",
        "tags": ["sophisme", "biais", "argumentation", "logique"],
        "q": "Quels sont les sophismes logiques les plus fréquents à éviter ?",
        "thought": "Citons l'homme de paille, l'attaque ad hominem, le faux dilemme et post hoc ergo propter hoc.",
        "a": (
            "Quatre sophismes majeurs faussent le raisonnement : l'**homme de paille** (déformer la thèse adverse), l'**ad hominem** (attaquer la personne plutôt que l'argument), "
            "le **faux dilemme** (réduire un spectre complexe à deux extrêmes) et **post hoc** (confondre corrélation temporelle et causalité)."
        ),
    },
    {
        "domain": "history",
        "key": "domain_hist_digital_birth",
        "tags": ["turing", "shannon", "ère numérique", "informatique"],
        "q": "Quel rôle Alan Turing et Claude Shannon ont-ils joué dans la naissance de l'ère numérique ?",
        "thought": "Croisons l'article de Turing (1936) sur la calculabilité universelle et ceux de Shannon (1937, 1948) sur les circuits booléens et le bit.",
        "a": (
            "En 1936, **Alan Turing** a prouvé qu'une seule machine universelle programmable peut exécuter tout algorithme calculable. "
            "Puis **Claude Shannon** a démontré que l'algèbre de Boole réalise tous les circuits logiques électroniques (1937) et a fondé en 1948 la **théorie de l'information**, définissant le *bit* et la capacité maximale des canaux bruités."
        ),
    },
    {
        "domain": "history",
        "key": "domain_hist_printing_science",
        "tags": ["gutenberg", "imprimerie", "révolution scientifique", "galilée", "newton"],
        "q": "Pourquoi l'imprimerie de Gutenberg a-t-elle accéléré la Révolution scientifique ?",
        "thought": "Relions la reproductibilité exacte des tables astronomiques, la baisse du coût du livre et la critique ouverte.",
        "a": (
            "Vers 1450, les caractères mobiles de **Gutenberg** ont divisé par dix le coût de diffusion du savoir et éliminé les erreurs de copie manuelle : "
            "les tables astronomiques, planches anatomiques et démonstrations de Copernic, Galilée et Newton ont pu être vérifiées, critiquées et enrichies simultanément dans toute l'Europe."
        ),
    },
    {
        "domain": "economics",
        "key": "domain_econ_inflation",
        "tags": ["inflation", "taux directeur", "banque centrale", "monnaie"],
        "q": "Quelle relation lie l'inflation, la masse monétaire et les taux directeurs ?",
        "thought": "Expliquons MV = PQ et le mécanisme du crédit bancaire.",
        "a": (
            "Lorsque la demande globale et la création monétaire croissent plus vite que la production réelle de biens et services, le niveau général des prix monte (**inflation**). "
            "En relevant ses **taux directeurs**, une banque centrale renchérit le crédit, encourage l'épargne et refroidit la surchauffe pour stabiliser la monnaie."
        ),
    },
    {
        "domain": "economics",
        "key": "domain_econ_comparative_advantage",
        "tags": ["ricardo", "avantage comparatif", "commerce", "coût d'opportunité"],
        "q": "Explique la théorie de l'avantage comparatif de David Ricardo.",
        "thought": "Montrons que le gain à l'échange dépend du coût d'opportunité relatif et non de l'avantage absolu.",
        "a": (
            "**David Ricardo** (1817) a démontré que même si un acteur est plus productif qu'un autre dans *tous* les domaines (avantage absolu), "
            "les deux gagnent à se spécialiser chacun là où leur **coût d'opportunité relatif** est le plus faible, puis à échanger : la production totale augmente mathématiquement."
        ),
    },
    {
        "domain": "cybersecurity",
        "key": "domain_sec_sym_asym_hash",
        "tags": ["chiffrement", "symétrique", "asymétrique", "aes", "ed25519", "sha256"],
        "q": "Quelle est la différence entre chiffrement symétrique, asymétrique et hachage cryptographique ?",
        "thought": "Comparons AES-GCM, RSA/X25519/Ed25519 et SHA-256/Argon2.",
        "a": (
            "- **Chiffrement symétrique** (`AES-256-GCM`, `ChaCha20-Poly1305`) : une même clé secrète ultra-rapide chiffre et déchiffre les données.\n"
            "- **Cryptographie asymétrique** (`X25519`, `Ed25519`, `RSA`) : une paire **clé publique / clé privée** permet d'échanger un secret sur un réseau ouvert ou de signer numériquement.\n"
            "- **Hachage** (`SHA-256`, `BLAKE2b`, `Argon2id`) : transformation irréversible garantissant l'intégrité ou le stockage sûr des mots de passe."
        ),
    },
    {
        "domain": "cybersecurity",
        "key": "domain_sec_sqli_xss",
        "tags": ["injection sql", "xss", "csrf", "owasp", "sécurité web"],
        "q": "Comment protéger efficacement une application contre les injections SQL et XSS ?",
        "thought": "Précisons les requêtes préparées paramétrées et l'échappement contextuel + CSP.",
        "a": (
            "Contre l'**injection SQL**, utilise systématiquement des **requêtes préparées paramétrées** (jamais de concaténation de chaînes SQL). "
            "Contre le **XSS** (*Cross-Site Scripting*), échappe toute donnée dynamique selon son contexte HTML/JS et applique un en-tête strict **`Content-Security-Policy` (CSP)**."
        ),
    },
    {
        "domain": "english_polymath",
        "key": "domain_en_mla_attention",
        "tags": ["attention", "mla", "transformer", "complexity", "kv cache"],
        "q": "Explain the time and memory complexity of standard Self-Attention versus Multi-Head Latent Attention (MLA).",
        "thought": "Compare O(L^2 d) compute and KV cache size 2*L*H*d_h vs compressed latent KV rank d_c.",
        "a": (
            "Standard causal **Self-Attention** takes `O(L² · d)` compute and stores `2 · n_heads · d_head` KV scalars per token. "
            "**Multi-Head Latent Attention (MLA)**—used inside my `ChronoMorph-LM` core—projects Keys and Values through a shared low-rank latent bottleneck `c_KV` (`d_c ≪ d_model`) with decoupled RoPE, drastically shrinking KV-cache memory while preserving multi-head expressivity."
        ),
    },
    {
        "domain": "english_polymath",
        "key": "domain_en_p_vs_np",
        "tags": ["p vs np", "complexity", "np-complete", "sat", "algorithms"],
        "q": "What is the P vs NP problem in computer science?",
        "thought": "Contrast polynomial-time verification (NP) with polynomial-time decision/search (P) and Cook-Levin SAT.",
        "a": (
            "**P** is the class of decision problems solvable in polynomial time `O(n^k)`, while **NP** is the class of problems whose candidate solutions can be *verified* in polynomial time. "
            "**P vs NP** asks whether quick verification always implies quick discovery (`P = NP`). By the Cook–Levin theorem, solving boolean `SAT` in polynomial time would solve every problem in `NP`."
        ),
    },
    {
        "domain": "conversation",
        "key": "domain_conv_rag_explain",
        "tags": ["rag", "document", "fichier", "upload", "pdf", "texte"],
        "q": "Comment puis-je te faire lire et analyser mes propres documents ?",
        "thought": "Expliquons le bouton Document RAG de l'interface et la commande /doc.",
        "a": (
            "Clique simplement sur **`📎 Document RAG`** au-dessus de la barre de message pour importer un fichier (`.md`, `.txt`, `.py`, `.csv`, `.json`). "
            "Je le découpe et l'indexe localement dans `EvolveMem`, puis tu peux m'interroger dessus directement ou via `/doc` avec citations des sources !"
        ),
    },
]
