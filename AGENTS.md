# AGENTS.md — metadata-asf

Ce document décrit le projet `metadata-asf`, ses conventions de code, son
périmètre et son plan de développement. Il est destiné à tout agent (humain ou IA)
qui contribue au dépôt.

---

## 1. Écosystème : deux dépôts indépendants

Le projet global est volontairement scindé en **deux dépôts Git indépendants**,
ayant chacun leur cycle de release, leur CI et leur licence. Ce fichier `AGENTS.md`
ne concerne que le **premier dépôt**.

| Dépôt | Rôle | Dépend de | Statut |
|---|---|---|---|
| `metadata-asf` | Collecte et export des **métadonnées** ASF (recherche, extraction, Parquet) | — | **Ce dépôt** — en développement |
| `fetch-asf` | **Téléchargement** des produits ASF (Earthdata, reprise, vérification) | `metadata-asf` (recherche) | Dépôt séparé — planifié plus tard |

### Frontière fonctionnelle

- **metadata-asf** : lecture seule, requêtes API légères, sortie Parquet (Ko–Mo).
- **fetch-asf** : authentification Earthdata, I/O lourd, sortie produits SAFE/HDF5 (Go–To).
- Aucun code de download, aucune dépendance à `earthaccess`/`requests`
  authentifié, aucun secret dans `metadata-asf`.
- Le catalogue produit par `metadata-asf` (fichiers Parquet) est le
  **contrat d'interface** entre les deux dépôts : `fetch-asf`
  lira ces Parquet pour savoir quoi télécharger.

> **Règle d'or :** si une fonctionnalité nécessite des identifiants
> Earthdata ou transfère plus de quelques Mo, elle n'a rien à faire dans ce dépôt.
> Elle appartient à `fetch-asf`.

---

## 2. Objectif du projet `metadata-asf`

Fournir une bibliothèque Python 3 et une CLI pour interroger l'API ASF et exporter
les métadonnées des acquisitions SAR dans des **fichiers Parquet journaliers**.

### Périmètre fonctionnel

- **Fournisseur** : ASF (Alaska Satellite Facility), via `asf_search`.
- **Missions** : conçu **multi-missions** dès le départ ; la
  première mission cible est **NISAR**.
- **Types de produits NISAR ciblés** :
  - **L1 — RSLC** (`Range-Doppler Single Look Complex`)
  - **L2 — GSLC** (`Geocoded Single Look Complex`)
- **Zone géographique** : océans par défaut (filtrage WKT configurable).
- **Sortie** : un fichier Parquet par jour, avec les colonnes suivantes :
  - `granule_id` (str) — filename / granuleName
  - `platform` (str) — ex. `NISAR`, `SENTINEL-1A`…
  - `geometry` (str WKT) — footprint
  - `start_time` (datetime UTC) — début d'acquisition
  - `stop_time` (datetime UTC) — fin d'acquisition
  - `polarization` (list[str]) — ex. `["HH", "HV"]`
  - `beam_mode` (str) — mode d'acquisition
  - `product_type` (str) — ex. `RSLC`, `GSLC`
  - `processing_level` (str) — `L1`, `L2`

### Hors périmètre (explicitement)

- Téléchargement de produits → `fetch-asf`.
- Authentification Earthdata → `fetch-asf`.
- Traitement scientifique des données SAR (InSAR, calibration, etc.).
- Autres niveaux NISAR (RIFG, RUNW, ROFF, GUNW, GCOV, GOFF, RRSD) — au moins dans un premier temps.

---

## 3. Conception multi-missions

L'API ASF expose plusieurs missions (`SENTINEL-1`, `ALOS-2`,
`SMAP`, `NISAR`, missions historiques…). L'architecture doit
permettre d'ajouter une mission **sans toucher au cœur du code**.

### Profils de mission

- Chaque mission est décrite par un **profil** (dataclass ou modèle Pydantic) :
  - `name` — identifiant CLI (ex. `NISAR`)
  - `asf_dataset` — valeur passée à `asf.search(dataset=...)`
  - `supported_products` — liste de `(processing_level, product_type)`
  - `default_ocean_wkt` — WKT par défaut (si pertinent)
  - `field_mapping` — correspondance entre les propriétés ASF et le schéma normalisé
- Les profils vivent dans `src/metadata_asf/profiles/` (un module par mission).
- Ajouter une mission = ajouter un module + un test + une entrée dans le registre.

### Registre de missions

```python
MISSIONS: dict[str, MissionProfile] = {
    "NISAR": nisar_profile,
}
```

---

## 4. Interface CLI

Le package expose une commande `metadata-asf` :

| Option | Type | Requis | Description |
|---|---|---|---|
| `--mission` | `str` | non | Mission cible (défaut : `NISAR`) |
| `--outputdir` | `Path` | oui | Répertoire de sortie des fichiers Parquet |
| `--log-verbosity` | `str` | non | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` (défaut : `INFO`) |
| `--date` | `str` | non | Date `YYYY-MM-DD` ou plage `YYYY-MM-DD:YYYY-MM-DD` |
| `--conf` | `Path` | non | Fichier de configuration YAML |

Exemple :

```bash
metadata-asf \
  --mission NISAR \
  --outputdir ./out \
  --log-verbosity INFO \
  --date 2025-01-01:2025-01-31 \
  --conf config.yaml
```

Priorité : `CLI > fichier --conf > valeurs par défaut du profil de mission`.

---

## 5. Stack technique

- **Python** ≥ 3.10
- **Runtime** :
  - `asf_search` — client officiel de l'API ASF (recherche)
  - `pandas` — manipulation tabulaire
  - `pyarrow` — écriture Parquet
  - `pydantic` (v2) — validation config & profils
  - `pyyaml` — lecture du fichier `--conf`
  - `tqdm` — barres de progression
  - `shapely` — conversion géométrie → WKT
  - `tenacity` — retry exponentiel sur appels API
  - `matplotlib` — figures du rapport HTML (rendues côté serveur, PNG intégrés en base64)
- **Dev** : `ruff`, `black`, `mypy`, `pytest`, `pytest-cov`, `pre-commit`
- **Logging** : module `logging` standard, sortie sur `stdout`.

> **Pas d'extras `[fetch]` ici.** Le download vit dans un autre
> dépôt. Toute dépendance liée à l'authentification ou au transfert de fichiers lourds
> est bannie de ce dépôt.

---

## 6. Conventions de code

- **Style** : `black` (line-length 100) + `ruff`.
- **Typage** : annotations obligatoires sur toutes les fonctions publiques ;
  `mypy --strict` doit passer.
- **Imports** : triés par `ruff` (isort intégré).
- **Logging** : `logger = logging.getLogger(__name__)` par module.
  Aucun `print()` sauf dans `cli.py` pour un usage strictement interactif.
- **Erreurs** : exceptions explicites, jamais de `except:` nu.
- **Docstrings** : format Google, en anglais (public API doc, README and Sphinx docs are in English).
- **Pas de code mort** : imports et variables inutilisés supprimés.
- **tqdm** : `tqdm(..., disable=not sys.stdout.isatty())` pour ne pas polluer les logs CI.
- **API publique stable** : les fonctions utilisées par `fetch-asf`
  (voir §7) doivent être documentées et versionnées selon SemVer.

---

## 7. Contrat d'interface avec `fetch-asf`

Même si `fetch-asf` est un dépôt séparé, il consommera `metadata_asf`.
Il faut donc définir un **contrat minimal, stable et testé**.

### API Python publique garantie

```python
from metadata_asf import search, extract, export
from metadata_asf.profiles import get_profile

# 1. Recherche — renvoie des objets ASFProduct bruts
products: list[asf.ASFProduct] = search.search(
    mission="NISAR",
    start=date(2025, 1, 1),
    end=date(2025, 1, 31),
    intersects_with=ocean_wkt,
    max_results=10_000,
)

# 2. Extraction — renvoie un DataFrame normalisé
df: pd.DataFrame = extract.to_dataframe(products, mission="NISAR")

# 3. Export — écrit les Parquet journaliers
written: list[Path] = export.write_daily_parquet(df, output_dir=Path("./out"))
```

### Schéma Parquet garanti

- Colonnes listées en §2, avec types stables.
- Nom de fichier : `{mission}_ocean_YYYYMMDD.parquet`.
- Compression `snappy`, `index=False`, UTC partout.
- Toute évolution de schéma = bump de version majeure + note de migration.

### Ce que `fetch-asf` ne doit PAS attendre de `metadata_asf`

- Résolution de credentials Earthdata.
- Téléchargement, reprise, checksum.
- Appels concurrents ou asynchrones.

---

## 8. Structure du dépôt

```text
metadata-asf/
├── AGENTS.md
├── README.md
├── Makefile
├── pyproject.toml
├── .pre-commit-config.yaml
├── config.example.yaml
├── src/
│   └── metadata_asf/
│       ├── __init__.py          # réexport de l'API publique
│       ├── cli.py               # argparse + orchestration main()
│       ├── config.py            # modèle Pydantic Config
│       ├── search.py            # enveloppe asf_search + retry
│       ├── extract.py           # ASFProduct[] -> DataFrame normalisé
│       ├── export.py            # DataFrame -> Parquet journalier
│       ├── profiles/
│       │   ├── __init__.py      # registre MISSIONS
│       │   ├── base.py          # MissionProfile (dataclass)
│       │   └── nisar.py         # profil NISAR (RSLC, GSLC)
│       └── utils.py             # dates, logging, WKT océanique
└── tests/
    ├── conftest.py
    ├── test_smoke.py            # squelette : à remplacer par les tests des étapes 2-8
```

---

## 9. Plan de développement

### Étape 1 — Bootstrap

- [x] Initialiser le dépôt (hatchling, CalVer via `hatch-vcs`, entry-point CLI,
      configuration ruff/black/mypy/pytest, `.pre-commit-config.yaml`).
- [x] Créer l'arborescence `src/metadata_asf/` et `tests/`.
- [x] Initialiser `README.md` (installation, usage, contrat d'interface).
- [ ] Finaliser la documentation Sphinx (`docs/`) au fil des étapes.

### Étape 2 — Profils de mission

- [x] `profiles/base.py` : modèle `MissionProfile`
      (`name`, `asf_dataset`, `supported_products`,
      `default_ocean_wkt`, `field_mapping`).
- [ ] `profiles/nisar.py` : profil NISAR avec
  `asf_dataset="NISAR"`, produits `[("L1", "RSLC"), ("L2", "GSLC")]`.
- [x] `profiles/__init__.py` : registre `MISSIONS` + `get_profile(name)`
  (registre vide, à compléter ; erreur claire si mission inconnue).
- [ ] Tests `test_profiles.py` (présence NISAR, profils inconnus → erreur claire).

### Étape 3 — Configuration et CLI

- [ ] `config.py` : modèle Pydantic `Config` :
  - `mission: str = "NISAR"`
  - `output_dir: Path`
  - `log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"`
  - `date_range: tuple[date, date] | None = None`
  - `ocean_wkt: str | None = None` (fallback : profil)
  - `processing_levels: list[str] | None = None` (fallback : profil)
  - `max_results: int = 10_000`
- [ ] `cli.py` : `argparse` avec `--mission`, `--outputdir`,
  `--log-verbosity`, `--date`, `--conf`. Fusion CLI > conf > profil.
- [ ] Configurer `logging` :
  `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`.
- [ ] Tests `test_config.py`.

### Étape 4 — Recherche ASF

- [ ] `search.py` :

```python
def search(
    mission: str,
    start: date,
    end: date,
    intersects_with: str | None = None,
    max_results: int = 10_000,
) -> list[asf.ASFProduct]:
    profile = get_profile(mission)
    return asf.search(
        dataset=profile.asf_dataset,
        processingLevel=profile.supported_products,
        intersectsWith=intersects_with or profile.default_ocean_wkt,
        start=start.isoformat(),
        end=end.isoformat(),
        maxResults=max_results,
    )
```

- [ ] Retry exponentiel via `tenacity` (3 tentatives, backoff 2×).
- [ ] Log `INFO` du nombre de résultats par appel.
- [ ] Support de `asf.search_generator()` si `max_results` atteint.
- [ ] Tests `test_search.py` avec `monkeypatch` sur `asf.search`.

### Étape 5 — Extraction des métadonnées

- [ ] `extract.py` : `to_dataframe(products, mission) -> pd.DataFrame`.
- [ ] Champs extraits selon le schéma normalisé (§2), via le
  `field_mapping` du profil de mission.
- [ ] `geometry` : `shapely.geometry.shape(product.geometry).wkt`.
- [ ] Dates en UTC : `pd.to_datetime(..., utc=True)`.
- [ ] `WARNING` pour chaque champ manquant, une seule fois par champ.
- [ ] Tests `test_extract.py` avec un `ASFProduct` factice.

### Étape 6 — Export Parquet journalier

- [ ] `export.py` : `write_daily_parquet(df, output_dir) -> list[Path]`.
  - Colonne `date` = `start_time.dt.date`.
  - Groupe par `date`, fichier `{mission}_ocean_YYYYMMDD.parquet`.
  - `engine="pyarrow"`, `compression="snappy"`, `index=False`.
- [ ] Boucle avec `tqdm(..., desc="Écriture Parquet")`.
- [ ] Log après chaque fichier : `Écrit {path} — {n} acquisitions ({ko:.1f} Ko)`.
- [ ] Résumé final :

```text
=== Résumé ===
Mission                : NISAR
Acquisitions totales   : X
Fichiers écrits        : Y
Dates couvertes        : D1 → D2
Erreurs                : Z
```

- [ ] Tests `test_export.py` sur DataFrame factice (vérifier nommage, colonnes, contenu).

### Étape 7 — Orchestration et erreurs

- [ ] `cli.py:main() -> int` : `0` succès, `1` erreur, `2` usage invalide.
- [ ] `--outputdir` : `mkdir(parents=True, exist_ok=True)`.
- [ ] `--date` invalide → message clair + `exit(2)`.
- [ ] Mission inconnue → liste des missions disponibles + `exit(2)`.
- [ ] Aucun résultat ASF → log `WARNING`, sortie `0`.

### Étape 8 — Tests, qualité, packaging

- [ ] Couverture ≥ 80 % (`pytest --cov`).
- [ ] `ruff check .`, `black --check .`, `mypy src/` verts.
- [ ] Test d'intégration `@pytest.mark.integration` (appel réel ASF, désactivé par défaut).
- [ ] `README.md` : exemples détaillés au fil des étapes.

---

## 10. Fichier de configuration exemple

`config.example.yaml` :

```yaml
mission: NISAR
output_dir: /data/nisar_ocean
log_level: INFO
date_range: ["2025-01-01", "2025-01-31"]
ocean_wkt: >
  POLYGON((-180 -90, 180 -90, 180 90, -180 90, -180 -90))
processing_levels: ["RSLC", "GSLC"]
max_results: 10000
```

---

## 11. Schéma du flux d'exécution

```text
┌────────────────────┐
│ Parse CLI + conf   │
└─────────┬──────────┘
          ▼
┌──────────────────────────┐
│ Résolution profil mission│
└─────────┬────────────────┘
          ▼
┌────────────────────┐
│ Config logging     │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Boucle sur dates   │──(tqdm)──▶┌──────────────────┐
└─────────┬──────────┘           │ asf.search()     │
          │                      │ RSLC + GSLC      │
          │                      └────────┬─────────┘
          │                               ▼
          │                      ┌──────────────────┐
          │                      │ extract → DF     │
          │                      └────────┬─────────┘
          ▼                               ▼
┌────────────────────┐           ┌──────────────────┐
│ Group by date      │◀──────────│ DataFrame global │
└─────────┬──────────┘           └──────────────────┘
          ▼
┌────────────────────┐
│ Write Parquet/jour │──▶ ./out/NISAR_ocean_YYYYMMDD.parquet
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Log résumé final   │
└────────────────────┘
                      ▼
              [fetch-asf lit ces Parquet]
```

---

## 12. Checklist avant commit

- [x] `ruff check .` passe
- [ ] `black --check .` passe
- [x] `mypy src/ tests/` passe en `--strict`
- [x] `pytest` passe (skeleton) — à atteindre ≥ 80 % de couverture
- [ ] Aucun `print()` résiduel (hors `cli.py`)
- [ ] Aucun import lié au download (`earthaccess`, `requests` authentifié…)
- [ ] Aucun secret, aucune clé API dans le code
- [x] Docstrings à jour sur les fonctions publiques modifiées
- [ ] Le schéma Parquet n'a pas changé sans bump de version
- [ ] `AGENTS.md` mis à jour si le périmètre change

---

## 13. Points d'attention

- **API ASF** : `processingLevel` est sensible à la casse
  (`RSLC`, `GSLC`).
- **NISAR** : au démarrage de la mission, le nom du dataset peut être
  `NISAR`, `NISAR_L1` ou `NISAR_L2` — centraliser
  cette valeur dans le profil de mission pour la corriger à un seul endroit.
- **Fuseaux horaires** : toujours UTC (`pd.to_datetime(..., utc=True)`).
- **WKT océanique** : le polygone mondial `-180..180 / -90..90`
  inclut les terres. Pour un vrai filtrage océanique, prévoir un WKT plus fin
  (ex. Natural Earth land polygons en négatif).
- **Volume** : NISAR produira de gros volumes ; utiliser
  `asf.search_generator()` si `max_results` est atteint.
- **Contrat d'interface** : toute évolution du schéma Parquet doit être
  répercutée dans `fetch-asf` — prévenir via un changelog clair et un bump
  SemVer.

---

## 14. Outils existants à réutiliser

Avant de réinventer, s'appuyer sur l'écosystème :

| Outil | Usage pour `metadata_asf` |
|---|---|
| [`asf_search`](https://github.com/asfadmin/Discovery-asf_search) | Backend de recherche ASF — dépendance principale. |
| [`itslive-metadata`](https://github.com/isce-framework/itslive-metadata) | Inspiration pour l'extraction des champs RSLC/GSLC (noms exacts des propriétés `ASFProduct`). |
| [`nisarqa`](https://github.com/isce-framework/nisarqa) | Inspiration pour la validation des métadonnées RSLC/GSLC. |
| [NISAR Cookbook](https://asfopensarlab.github.io/NISAR_Cookbook/) | Patterns de recherche NISAR, à reproduire dans nos tests d'intégration. |
| [`nisar-workflows`](https://github.com/isce-framework/nisar-workflows) | À surveiller pour les évolutions du format NISAR et des workflows communautaires. |

**Ce que `metadata_asf` apporte en propre** : la combinaison
<em>recherche multi-missions + filtrage océanique + export Parquet journalier</em>,
qui n'existe dans aucun outil existant à ce jour.
