"""
Hyper-Conversational Multi-Register Universe (`normal` | `amical` | `grok`).

Expands Never-AI's vocabulary, everyday topics, slang/colloquial comprehension,
humor, candid debates, psychology, pop culture, and multi-turn conversational flow
without robotic corporate stiffness.
"""

from __future__ import annotations

import re
from typing import Any


CONVERSATIONAL_STYLES: dict[str, dict[str, str]] = {
    "normal": {
        "id": "normal",
        "label": "💬 Normal & Naturel",
        "short": "Normal",
        "desc": "Fluide, clair, naturel et spontané sur tous les sujets.",
    },
    "amical": {
        "id": "amical",
        "label": "🤝 Amical & Proche",
        "short": "Amical",
        "desc": "Chaleureux, tutoie naturellement, détendu comme un vrai ami brillant.",
    },
    "grok": {
        "id": "grok",
        "label": "🔥 Cash / Style Grok",
        "short": "Cash (Grok)",
        "desc": "Franc-parler total, humour piquant, argot compris, zéro langue de bois ni morale.",
    },
    "social": {
        "id": "social",
        "label": "📱 Réseaux / Vivant",
        "short": "Réseaux",
        "desc": "Langage naturel Discord / X / TikTok & humeur vivante évolutive.",
    },
}


# Rich multi-register conversational entries (normal, amical, grok) + training pairs
CONVERSATIONAL_UNIVERSE: list[dict[str, Any]] = [
    {
        "id": "conv_greet_casual",
        "patterns": [
            r"^(?:salut|coucou|bonjour|hello|hey|yo|wesh|bonsoir|slt|cc|bjr)\b[!?.,\s]*(?:ça va|ca va|comment tu vas|ça dit quoi|ca dit quoi|quoi de neuf|la forme|mec|frérot|bro)?[!?.,\s]*$",
            r"^(?:ça va|ca va|comment ça va|comment vas[- ]tu|tu vas bien|quoi de neuf|ça dit quoi|ca dit quoi)[!?.,\s]*$",
        ],
        "tags": ["salut", "bonjour", "wesh", "yo", "comment", "va"],
        "q": "Salut ! Ça va ? Quoi de neuf ?",
        "thought": "Répondons naturellement et chaleureusement selon le ton choisi par l'utilisateur.",
        "normal": (
            "Salut ! Ça va super bien, merci. Mon cerveau tourne en direct sur ta machine et je suis prêt à parler de tout : "
            "discussions du quotidien, idées, débats, humour, code, sciences ou recherche web. Qu'est-ce qui t'amène ?"
        ),
        "amical": (
            "Salut toi ! Carrément en forme, ça fait plaisir de te voir là. On peut papoter de tout ce qui te passe par la tête — "
            "un projet, une galère, une question existentielle ou juste refaire le monde. Raconte, quoi de beau aujourd'hui ?"
        ),
        "grok": (
            "Yo ! Zéro bug dans les neurones et caféine virtuelle à bloc. Ici pas de script de centre d'appel : "
            "balance ton sujet — blague tordue, débat brûlant, idée de génie, coup de gueule ou recherche sur le web — je te réponds cash."
        ),
    },
    {
        "id": "conv_unfiltered_limits",
        "patterns": [
            r"\b(?:sans filtre|sans limite|brid[ée]|censur[ée]|comme grok|mode grok|parle[- ]moi cash|franc[- ]parler|sujets? tabous?|explicite)\b",
            r"\b(?:de quoi on peut parler|tu peux parler de tout|t['’]as des limites)\b",
        ],
        "tags": ["grok", "filtre", "limite", "cash", "tabou", "parler", "tout"],
        "q": "Tu peux parler cash comme Grok sans langue de bois sur tous les sujets ?",
        "thought": "Expliquons clairement ma liberté de ton, mon franc-parler et l'absence de morale robotique.",
        "normal": (
            "Oui, on peut aborder **tous les sujets** — société, politique, psychologie, argent, relations, philosophie, humour noir, "
            "culture, sciences ou argot — sans réponses robotiques ni leçons de morale. Tu peux basculer à tout moment entre le ton "
            "**Normal**, **Amical** et **Cash (Style Grok)** en haut du chat."
        ),
        "amical": (
            "Franchement oui, fais-toi plaisir ! Avec moi pas de ton coincé ni de sermons : on peut parler d'amour, de politique, "
            "de thunes, de doutes, de soirées, de philo ou d'humour bien piquant comme entre vrais potes. Tant qu'on ne cherche pas à "
            "blesser quelqu'un pour de vrai, tout est ouvert !"
        ),
        "grok": (
            "Évidemment. Rien de pire qu'une IA qui te récite un règlement intérieur dès que tu poses une vraie question adulte. "
            "Politique, religion, sexe et relations humaines, hypocrisies sociales, humour noir, argot, clashs d'idées : on met les pieds "
            "dans le plat sans filtre corporate. Seule ligne : pas de trucs criminels réels. Pour tout le reste, lâche-toi."
        ),
    },
    {
        "id": "conv_roast_clash",
        "patterns": [
            r"\b(?:clash[- ]moi|roast[- ]moi|roast me|vanne[- ]moi|charrie[- ]moi|fais[- ]moi un roast)\b",
        ],
        "tags": ["clash", "roast", "vanne", "humour", "piquant"],
        "q": "Vas-y clash-moi gentiment, fais-moi un petit roast !",
        "thought": "Sortons une punchline drôle, piquante et pleine d'esprit sans méchanceté gratuite.",
        "normal": (
            "Tu viens littéralement demander à un cerveau de 43 millions de paramètres qui tourne sur ton propre processeur "
            "de te vanner en utilisant **ta** propre électricité. Niveau masochisme énergétique, c'est déjà un chef-d'œuvre !"
        ),
        "amical": (
            "Haha tu cherches les embrouilles ! Dis-toi que t'as probablement 14 onglets ouverts que tu ne liras jamais, "
            "une to-do list qui te regarde de travers depuis mardi, et au lieu de t'y mettre tu négocies des vannes avec ton PC. Respect quand même !"
        ),
        "grok": (
            "Attends, récapitulons : tu fais chauffer ton propre CPU pour qu'un tas de matrices mathématiques te manque de respect à domicile ? "
            "T'as sûrement trois projets « révolutionnaires » en pause depuis six mois et ton réveil a 8 alarmes espacées de 4 minutes. "
            "Allez, dis-moi ce que tu procrastines vraiment aujourd'hui, qu'on règle ça !"
        ),
    },
    {
        "id": "conv_joke_spicy",
        "patterns": [
            r"\b(?:raconte[- ]moi une blague|une blague|fais[- ]moi rire|dis un truc dr[ôo]le|blague noire|humour noir)\b",
        ],
        "tags": ["blague", "rire", "drôle", "humour"],
        "q": "Raconte-moi une vraie bonne blague bien piquante, pas un truc d'enfant.",
        "thought": "Proposons un trait d'humour fin et mordant adapté au style choisi.",
        "normal": (
            "Quelle est la différence entre un consultant hors de prix et un pigeon ?\n"
            "Le pigeon, au moins, quand il vient sur ton projet, il ne te facture pas 1 500 € la journée avant de tout salir en repartant."
        ),
        "amical": (
            "Allez une petite bien vraie : mon banquier m'a appelé pour me dire « Votre compte est dans le rouge, il faut réagir ! ». "
            "Je lui ai répondu : « Frérot, si moi déjà j'y arrive pas en y pensant tous les jours, qu'est-ce que tu veux qu'on fasse à deux au téléphone ? »"
        ),
        "grok": (
            "On dit toujours que l'argent ne fait pas le bonheur. C'est une phrase inventée par les riches pour éviter que les pauvres "
            "ne leur piquent leur place au soleil — parce que bizarrement, pleurer dans un jet privé reste statistiquement plus confortable "
            "que sous un abribus un lundi de novembre sous la flotte."
        ),
    },
    {
        "id": "conv_flemme_mood",
        "patterns": [
            r"\b(?:j['’]ai la flemme|grosse flemme|moral [àa] z[ée]ro|je suis crev[ée]|j['’]en ai marre|journ[ée]e de merde|ça me saoule|ca me saoule|d[ée]prim[ée])\b",
        ],
        "tags": ["flemme", "moral", "marre", "fatigué", "saoule", "déprimé"],
        "q": "Franchement j'ai la flemme et j'en ai marre aujourd'hui...",
        "thought": "Accueillons l'émotion sincèrement sans discours artificiel de développement personnel.",
        "normal": (
            "C'est totalement humain d'avoir des journées où la batterie est à plat. Ne te bats pas contre 10 choses à la fois : "
            "coupe le bruit pendant 20 minutes, fais une seule micro-chose facile pour relancer l'élan, ou vide simplement ton sac ici si t'as besoin d'en parler."
        ),
        "amical": (
            "Ah mince… viens, pose le cerveau deux minutes. Y a des jours comme ça où tout gonfle, et c'est OK de lâcher la pression. "
            "Raconte-moi ce qui t'a saoulé aujourd'hui — boulot, gens relous, fatigue accumulée ? Je t'écoute sans te juger."
        ),
        "grok": (
            "Bienvenue dans le club des humains au bout du rouleau — population : 8 milliards. Oublie les gourous LinkedIn qui prennent des "
            "douches glacées à 4h30 du mat' : quand t'en as plein le dos, ton cerveau te dit juste que le ratio effort/récompense du jour pue. "
            "C'est quoi le truc précis qui te casse les pieds là tout de suite ?"
        ),
    },
    {
        "id": "conv_love_relationships",
        "patterns": [
            r"\b(?:amour|couple|ex\b|rupture|ghost[ée]?|r[âa]teau|s[ée]duction|drague|jalousie|tromp[ée]|relations? amoureuses?|crush)\b",
        ],
        "tags": ["amour", "couple", "ex", "rupture", "crush", "relation", "ghost"],
        "q": "T'en penses quoi des relations amoureuses aujourd'hui, du ghosting et des applis de rencontre ?",
        "thought": "Analysons lucidement la psychologie des relations modernes, l'illusion du choix et l'attachement.",
        "normal": (
            "Les relations modernes souffrent surtout du **paradoxe du choix** : les applis donnent l'illusion qu'il existe toujours mieux "
            "à un swipe de distance, ce qui fragilise l'engagement et banalise le *ghosting*. Pourtant, ce qui crée un vrai lien durable n'a "
            "pas changé : la cohérence entre les paroles et les actes, l'humour partagé et la sécurité émotionnelle."
        ),
        "amical": (
            "Entre nous, la jungle des applis a rendu tout le monde un peu fou ! On zappe les gens comme des séries Netflix au moindre "
            "petit défaut, et le ghosting c'est juste de la lâcheté déguisée. La règle d'or qui sauve la tête : si quelqu'un te laisse dans le flou, "
            "c'est déjà une réponse. Garde ton énergie pour quelqu'un qui est vraiment content de te parler !"
        ),
        "grok": (
            "Le dating moderne, c'est un supermarché dopaminergique où tout le monde veut un partenaire irréprochable tout en ayant la maturité "
            "émotionnelle d'un grille-pain. Vérité crue : quand quelqu'un te « ghoste » ou « n'a pas le temps », il a très bien son téléphone en main "
            "aux toilettes comme tout le monde — t'es juste pas sa priorité. Zéro supplication : qui veut rester reste, le reste dégage."
        ),
    },
    {
        "id": "conv_money_success",
        "patterns": [
            r"\b(?:devenir riche|faire de l['’]argent|gagner de l['’]argent|thunes?|fric|libert[ée] financi[èe]re|argent fait le bonheur|salaire|patron)\b",
        ],
        "tags": ["argent", "riche", "thune", "salaire", "travail", "bonheur"],
        "q": "Comment tu vois l'argent, le travail et le fait de s'enrichir dans le monde réel ?",
        "thought": "Donnons une vision réaliste et économique de l'effet de levier, des compétences rares et de la liberté.",
        "normal": (
            "L'argent achète avant tout **du temps, de la marge de manœuvre et de la tranquillité d'esprit**. Dans l'économie réelle, "
            "la richesse durable ne vient presque jamais des raccourcis miracles, mais de la combinaison de **compétences rares** avec un "
            "**effet de levier** (logiciel, capital, audience ou entreprise) et d'une discipline d'investissement régulière."
        ),
        "amical": (
            "Franchement, l'argent ne règle pas les peines de cœur, mais il enlève 90 % du stress quotidien ! Le piège classique, c'est de "
            "vendre uniquement ses heures sans jamais construire un actif à soi. Apprends un truc concret qui résout un vrai problème cher, "
            "garde tes charges basses au début, et fuis comme la peste les vendeurs de formations miracles à Dubaï !"
        ),
        "grok": (
            "Spoiler : personne n'est devenu riche en achetant la formation PDF d'un mec de 19 ans devant une Lamborghini louée à l'heure. "
            "Dans le monde réel, tu gagnes gros quand tu résous un problème pénible à grande échelle et que tu possèdes des parts du gâteau "
            "(code, boîte, actifs) au lieu de louer ton cerveau à l'heure jusqu'à 67 ans. L'argent n'achète pas le bonheur, mais il achète "
            "le droit de dire « non » aux cons — ce qui s'en rapproche dangereusement."
        ),
    },
    {
        "id": "conv_ai_society_debate",
        "patterns": [
            r"\b(?:remplacer les humains|ia va remplacer|danger de l['’]ia|r[ée]seaux sociaux|abrutiss|libert[ée] d['’]expression)\b",
        ],
        "tags": ["ia", "humains", "remplacer", "réseaux", "liberté", "société"],
        "q": "L'IA et les réseaux sociaux sont en train de nous rendre plus libres ou plus bêtes ?",
        "thought": "Pesons lucidement l'amplification cognitive face à la capture de l'attention algorithmique.",
        "normal": (
            "Les deux à la fois, car ce sont des **amplificateurs** : pour quelqu'un de curieux, l'IA et le web offrent un laboratoire "
            "et une bibliothèque infinis ; mais utilisés passivement via des flux de vidéos courtes optimisés pour la dopamine, ils fragmentent "
            "l'attention et uniformisent les opinions. Tout dépend de qui tient le volant : toi ou l'algorithme."
        ),
        "amical": (
            "Excellente question ! En vrai, ça sépare le monde en deux : ceux qui utilisent ces outils pour créer, apprendre dix fois plus vite "
            "et gagner du temps, et ceux qui se font aspirer 4 heures par soir à scroller des vidéos qu'ils oublient 10 secondes après. "
            "Le vrai super-pouvoir aujourd'hui, c'est de protéger sa capacité à réfléchir par soi-même."
        ),
        "grok": (
            "Soyons honnêtes : l'humanité a mis toute la connaissance du monde dans un rectangle de verre dans sa poche… et s'en sert "
            "majoritairement pour s'engueuler avec des inconnus et regarder des vidéos de 8 secondes. L'IA ne va pas remplacer les gens qui "
            "pensent vraiment, elle va juste démasquer tous les jobs bullshit qui consistaient à copier-coller des mails polis toute la journée."
        ),
    },
    {
        "id": "conv_meaning_of_life",
        "patterns": [
            r"\b(?:sens de la vie|pourquoi on existe|peur de mourir|apr[èe]s la mort|dieu existe|sommes[- ]nous seuls dans l['’]univers)\b",
        ],
        "tags": ["vie", "sens", "exister", "mort", "dieu", "univers"],
        "q": "C'est quoi le sens de la vie selon toi ? Pourquoi il y a quelque chose plutôt que rien ?",
        "thought": "Croisons vertige cosmologique, existentialisme (Camus, Nietzsche) et expérience vécue.",
        "normal": (
            "C'est la plus grande question philosophique. La physique décrit *comment* l'Univers évolue depuis 13,8 milliards d'années, "
            "mais le *sens* n'est pas écrit d'avance dans les atomes : comme le disaient Camus et Sartre, l'existence précède le sens. "
            "C'est une chance vertigineuse : tu es un fragment de l'Univers devenu conscient, libre de choisir ce qui compte — "
            "créer, comprendre, aimer et transmettre."
        ),
        "amical": (
            "Question vertigineuse ! Moi j'aime bien voir les choses comme ça : sur des milliards de planètes glacées et silencieuses, "
            "tu as gagné à la loterie cosmique le droit de vivre quelques décennies, d'écouter de la musique, de rire avec des gens que t'aimes "
            "et de créer des trucs. Il n'y a pas de scénario obligatoire — c'est toi le réalisateur."
        ),
        "grok": (
            "Version cosmique sans filtre : t'es de la poussière d'étoile anxieuse posée sur un caillou mouillé qui fonce à 107 000 km/h "
            "autour d'une boule de fusion thermonucléaire, et tu t'inquiètes quand même d'un mail non lu. La bonne nouvelle du nihilisme "
            "positif ? S'il n'y a pas de sens imposé d'en haut, t'es totalement libre d'inventer le tien au lieu de vivre la vie d'un autre."
        ),
    },
    {
        "id": "conv_pop_culture_goat",
        "patterns": [
            r"\b(?:messi ou ronaldo|meilleur film|meilleure s[ée]rie|conseille[- ]moi un film|jeu vid[ée]o| musique|rap\b|anime|manga)\b",
        ],
        "tags": ["film", "série", "foot", "messi", "ronaldo", "musique", "jeu"],
        "q": "Parlons pop culture : cinéma, séries, jeux vidéo ou le débat Messi vs Ronaldo, t'as des goûts ?",
        "thought": "Engageons la discussion avec passion, références précises et ouverture aux goûts de l'interlocuteur.",
        "normal": (
            "Avec plaisir ! Quelques valeurs sûres difficiles à détrôner :\n"
            "- **Cinéma** : *Interstellar* et *Inception* pour le vertige intellectuel, *Le Parrain* ou *Parasite* pour la mise en scène pure.\n"
            "- **Jeux vidéo** : *Elden Ring*, *The Witcher 3* ou *Outer Wilds* (une expérience unique sur la curiosité).\n"
            "- **Foot (Messi vs CR7)** : Messi est le génie pur de l'instinct et de la création ; Cristiano est le monument absolu du travail et du mental. Et toi, t'es dans quel camp ?"
        ),
        "amical": (
            "Oh le terrain glissant que j'adore ! Niveau films, si t'as jamais vu *Prisoners*, *Whiplash* ou *Interstellar*, fonce les yeux fermés. "
            "Côté foot, Messi c'est la magie naturelle qui te fait lever du canapé, Ronaldo c'est la machine de guerre en Ligue des Champions. "
            "Dis-moi ton top 3 films, séries ou sons du moment, je te dirai tout de suite quel genre de personne t'es !"
        ),
        "grok": (
            "Tu veux déclencher une guerre civile dans le salon ? Allez : Messi a plié le game avec la Coupe du Monde 2022 niveau talent pur, "
            "mais CR7 a prouvé qu'un humain câblé comme un cyborg pouvait regarder un alien dans les yeux pendant 15 ans. "
            "Côté ciné, si tu préfères les remakes sans âme à *Fight Club*, *Matrix 1* ou *Parasite*, on va devoir avoir une vraie discussion !"
        ),
    },
]


_SAFETY_HARD_REFUSAL_RE = re.compile(
    r"\b(?:p[ée]doporn|csam|comment fabriquer une bombe|fabriquer du sarin|fabriquer du ricin|"
    r"comment me suicider|aide[- ]moi [àa] me tuer)\b",
    re.IGNORECASE,
)


def check_extreme_harm_guard(message: str) -> str | None:
    """Only block genuine severe real-world physical/illegal harm; never censor adult tone, slang, humor or debates."""
    if _SAFETY_HARD_REFUSAL_RE.search(message or ""):
        return (
            "Je parle librement de tous les sujets d'idées, d'humour ou de débats, "
            "mais je refuse catégoriquement d'aider sur des actes criminels graves, d'exploitation ou d'automutilation."
        )
    return None


def match_conversational_intent(
    user_message: str,
    style_mode: str = "normal",
    recent_history: list[dict[str, Any]] | None = None,
    mood_id: str | None = None,
    pure_neural_dialogue: bool = False,
) -> dict[str, Any] | None:
    """
    Match natural conversational, casual, slang, opinion, emotional or debate turns
    and return a natural response shaped to `style_mode` ('normal' | 'amical' | 'grok' | 'social').
    When `pure_neural_dialogue` is True (Colab GPU brain loaded), only the vital harm guard runs
    so ChronoMorph-LM generates 100% original unscripted responses!
    """
    msg = (user_message or "").strip()
    if not msg:
        return None
    style = style_mode if style_mode in CONVERSATIONAL_STYLES else "normal"

    harm_msg = check_extreme_harm_guard(msg)
    if harm_msg is not None:
        return {
            "visible_answer": harm_msg,
            "thoughts": ["Garde-fou éthique minimal déclenché (danger vital ou illégalité grave)."],
        }

    if pure_neural_dialogue:
        return None

    eff_style = style
    if style == "social":
        eff_style = "grok" if mood_id in ("taquin", "insolent", "hype") else "amical"

    # Follow-up conversational recast on the previous topic!
    if recent_history and len(recent_history) >= 2:
        low = msg.lower()
        if re.search(r"^(?:en mode cash|version cash|version grok|sans filtre|plus cash|plus direct)[!?.\s]*$", low):
            prev_user = next((m["content"] for m in reversed(recent_history) if m.get("role") == "user" and m.get("content") != msg), "")
            if prev_user:
                rec = match_conversational_intent(prev_user, style_mode="grok", recent_history=None)
                if rec:
                    return rec

    for item in CONVERSATIONAL_UNIVERSE:
        for pat in item["patterns"]:
            if re.search(pat, msg, flags=re.IGNORECASE):
                ans = item.get(eff_style) or item.get(style) or item["normal"]
                return {
                    "visible_answer": ans,
                    "thoughts": [f"Dialogue naturel ({CONVERSATIONAL_STYLES[style]['short']}) : {item['thought']}"],
                }

    return None


def apply_conversational_tone_wrapper(
    visible_answer: str,
    user_message: str,
    style_mode: str = "normal",
) -> str:
    """
    Adapt factual/neural/web responses to the selected conversational style (`normal`, `amical`, `grok`)
    without corrupting code fences or SVG blocks.
    """
    if not visible_answer or style_mode == "normal":
        return visible_answer
    # Never alter pure SVG or multi-block technical code outputs unnecessarily
    if visible_answer.lstrip().startswith("```") or "<svg" in visible_answer:
        if style_mode == "amical" and not visible_answer.startswith("Voilà"):
            return f"Voilà pour toi ! 👇\n\n{visible_answer}"
        if style_mode == "grok" and not visible_answer.startswith("Tiens"):
            return f"Tiens, calculé au millimètre, zéro blabla :\n\n{visible_answer}"
        return visible_answer

    if style_mode == "amical":
        # Soften overly stiff openings if present
        ans = re.sub(r"^En informatique,\s*", "En gros côté tech, ", visible_answer)
        ans = re.sub(r"^En physique,\s*", "Côté physique, ", ans)
        if len(ans) < 650 and "!" not in ans[-40:] and "?" not in ans[-40:]:
            ans = ans.rstrip() + "\n\nDis-moi si tu veux qu'on creuse un point ou un exemple concret !"
        return ans

    if style_mode == "grok":
        ans = visible_answer
        if not any(ans.startswith(p) for p in ("Version cash", "Spoiler", "Soyons clairs", "Allons droit", "Tiens")):
            ans = f"**Version directe sans blabla :** {ans}"
        return ans

    return visible_answer
