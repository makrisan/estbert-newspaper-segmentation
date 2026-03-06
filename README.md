# iaib

# Keelemudelipõhine ajaleheartiklite eraldamine ja temaatiline analüüs
Autorid: Maria Kristina Andranikyan, Greteli Mittal
Juhendaja: Innar Liiv, PhD, Tallinna Tehnikaülikool (TalTech), infotehnoloogia teaduskond

## Projekti kirjeldus

Töö eesmärk on sarnaste katsetuste tegemine vabakasutuses olevatel eestikeelsetel artikliteks segmenteerimata ajalehtedel. Kõige parema mudeli tööd tuleb demonstreerida andmeanalüüsiga, kus ajalehest eraldatakse artiklid vabalt valitud teemal ja hinnatakse nende terviklikkust.

### Väljund

- **Peenhäälestatud mudel** ajalehtede artikliteks jaotamiseks (BERT-põhine). Mudel peab olema vabavaraline.
- Artikleid võib eraldada nii ühe lehekülje piires kui ka proovida kokku tuua mitmele lehele jaotunud artiklite teksti.
- **Andmeanalüüs** töö käigus valminud mudeli demonstreerimiseks.

## Käivitusjuhised

### 1. Repositooriumi kloneerimine

```bash
git clone https://gitlab.cs.taltech.ee/grmitt/iaib.git
cd iaib
git checkout main
```

### 2. Virtuaalkeskkonna loomine ja aktiveerimine

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

Sõltuvusi saab ükshaaval installida, kui requirements ei tööta.

```bash
pip install [nimi]
```

### 4. Projekti struktuur

```
IAIB/
│
├─ src/                    # Python lähtekood
├─ notebooks/              # Jupyter notebooks
├─ data/                   # Andmestikud (gitignored)
├─ models/                 # Treenitud mudelid / checkpointid
├─ scripts/                # Abiskriptid (treening, evalueerimine, jne)
├─ venv/                   # Python virtuaalkeskkond (gitignored)
├─ requirements.txt        # Python sõltuvused
├─ .gitignore
└─ README.md
```

## Põhiline töövoog

1. **Andmete ettevalmistamine** - ajalehtede laadimine ja eeltöötlus
2. **Mudeli treenimine** - BERT mudeli peenhäälestus artiklite segmenteerimiseks
3. **Evalueerimine** - mudeli jõudluse hindamine
4. **Analüüs** - temaatiline artiklite eraldamine ja analüüs

## Kasutatavad tehnoloogiad

- **PyTorch** - mudeli treenimine
- **Transformers** (Hugging Face) - BERT mudel
- **PyMuPDF** - PDF-ide töötlemine
- **Pandas, NumPy** - andmeanalüüs
- **scikit-learn** - mudelite evalueerimine