# iaib

# Keelemudelipõhine ajaleheartiklite eraldamine ja temaatiline analüüs

Autorid: Maria Kristina Andranikyan, Greteli Mittal  
Juhendaja: Innar Liiv, PhD — Tallinna Tehnikaülikool, infotehnoloogia teaduskond

## Projekti kirjeldus

Projekt peenhäälestab eestikeelse BERT-mudeli (EstBERT) digiteeritud ajalehtede 
segmenteerimiseks eraldi artikliteks, kasutades ainult tekstilist konteksti ilma 
visuaalse paigutuse teabeta. Teostatud koostöös Eesti Rahvusraamatukoguga (RaRa).

### Projekti struktuur

```
iaibLoputoo/
│
├── ai_lab_results/    # Treenitud mudel ja checkpointid (TalTech AI-labor, GPU)
├── analyze/           # Temaatiline analüüs välise testandmestiku põhjal
├── data/              # Andmestikud — nii segmenteeritud kui segmenteerimata (gitignored)
├── scripts/           # Peamised skriptid: treenimine, hindamine, rakendamine
├── src/               # Abikood: seadistus, andmete puhastus, eeltöötlus
├── loputoo/           # Lõputöö dokumendid
├── requirements.txt
├── .gitignore
└── README.md
```
## Käivitusjuhised

### 1. Repositooriumi kloneerimine

```bash
git clone https://gitlab.cs.taltech.ee/grmitt/iaib.git
cd iaib
```

### 2. Virtuaalkeskkonna loomine

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Sõltuvuste installeerimine

```bash
pip install -r requirements.txt
```

## Töövoog

Kogu töövoogu käivitab `scripts/run_pipeline.py`, mis jooksutab järjest:

| Samm | Skript | Kirjeldus |
|------|--------|-----------|
| 1 | `build_large_dataset.py` | Kolmikandmestiku koostamine segmenteeritud failidest |
| 2 | `split_dataset.py` | Jagamine train/val/test osadeks (80/10/10) |
| 3 | `train.py` | EstBERT peenhäälestamine kaalutud kaotusega |
| 4 | `evaluate.py` | Läve optimeerimine ja F1/täpsus/saagis hindamine |
| 5 | `inference.py` | Liugaknaga rakendamine segmenteerimata tekstidele |

```bash
python scripts/run_pipeline.py
```

## Tehnoloogiad

| Tehnoloogia | Versioon | Kasutus |
|-------------|----------|---------|
| Python | 3.11 | — |
| PyTorch | — | Mudeli treenimine |
| Transformers (Hugging Face) | — | EstBERT mudel |
| scikit-learn | — | Hindamine (F1, segadusmaatriks) |

## Tulemused

- **F1-skoor treeningandmestiku testandmestikul:** 0.65  
- **F1-skoor välisel testandmestikul:** 0.74