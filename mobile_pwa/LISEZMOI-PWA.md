# Never-AI v4.2 — Application Complète PWA Auto-Évolutive

Cette archive contient l'application **Never-AI (`v4.2.0` · Cortex 3.89B Souverain Sans Limite)** au format **Progressive Web App (PWA)** plein écran avec **double mise à jour automatique** (Service Worker + OTA GitHub sans réinstallation).

## 1. Installation directe en 1 clic (Android, iOS, Windows, macOS, Linux)

Depuis votre serveur Never-AI Studio (HTTPS ou `http://localhost:8000/?pwa=1`) :

- **Android (Chrome / Edge / Brave)** : ouvrez l'URL hors cadre -> appuyez sur **Installer l'App PWA** (ou menu `⋮` -> *Installer l'application*). Compatible Family Link (aucun APK tiers requis).
- **iPhone / iPad (Safari)** : ouvrez l'URL -> bouton **Partager** -> **Sur l'écran d'accueil**.
- **Windows / macOS / Linux (Chrome / Edge)** : cliquez sur **Installer l'App PWA** ou sur l'icône d'installation dans la barre d'adresse pour obtenir une fenêtre bureau dédiée plein écran.

## 2. Comment fonctionne la mise à jour automatique ?

Une fois installée, vous n'avez **plus jamais besoin de réinstaller l'application** :

1. **Mise à jour automatique de l'interface (Service Worker `v4.2+`)** : à chaque ouverture, retour au premier plan ou toutes les 3 minutes, le Service Worker vérifie en `no-store` la dernière build, purge automatiquement l'ancien cache et recharge l'application dès qu'aucun message n'est en cours de génération.
2. **Mise à jour automatique du moteur (OTA GitHub)** : l'application interroge automatiquement `github.com/mouusieur/never-ai`, télécharge et applique à chaud les nouveaux commits tout en **préservant 100 % de vos discussions, souvenirs (`memory_store/`) et poids (`checkpoints/`)**.

## 3. Déploiement autonome du dossier `NeverAI-PWA`

Vous pouvez héberger les fichiers de ce dossier sur n'importe quel hébergeur statique HTTPS (GitHub Pages, Cloudflare Pages, Vercel, Netlify) ou en local (`python -m http.server 8080`) et connecter votre serveur Cortex 3.89B via :

- Onglet **Applications & Mises à jour** -> champ **Serveur Cortex distant**, ou
- Paramètre d'URL direct : `index.html?server=https://votre-serveur-never-ai.app`
