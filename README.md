# Live BirdNET vogelherkenning

Een lokale Flask-webapp voor het herkennen van vogelgeluiden via een Mac-microfoon. De app gebruikt [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer) voor de analyse en toont detecties, confidence-scores, audiofragmenten en spectrogrammen in een eenvoudige webinterface.

De applicatie is bedoeld voor lokaal gebruik en onderzoek. Een detectie is een modelvoorspelling en geen gegarandeerde soortbepaling.

## Mogelijkheden

- Live audio analyseren in blokken van drie seconden.
- Vogelgeluiden filteren op geografische locatie en minimale confidence.
- De volledige sessie lokaal opslaan als MP3, M4A of WAV.
- Detectiefragmenten en spectrogrammen lokaal bewaren.
- Bestaande audiobestanden uploaden voor analyse.
- Nederlandse vogelnamen tonen via een uitbreidbare mapping.

## Installatie

Gebruik bij voorkeur Python 3.11. De globale `python3` op deze Mac is 3.14 en is niet geschikt voor BirdNET.

Vereist zijn daarnaast `uv`, Homebrew, `portaudio` en `ffmpeg`. De eerste installatie en de eerste analyse kunnen extra modelbestanden downloaden.

```bash
brew install portaudio ffmpeg
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
pip install -r requirements.txt
```

Start de app:

```bash
source .venv/bin/activate
python app.py
```

Open daarna:

```text
http://127.0.0.1:5055
```

## Opslag

De volledige sessie-opname komt standaard als compacte MP3 in:

```text
data/recordings/
```

Detectiefragmenten boven de drempelwaarde komen in:

```text
data/detections/
```

Spectrogram-afbeeldingen per detectie komen in:

```text
data/spectrograms/
```

Geüploade audiobestanden worden lokaal bewaard in:

```text
data/uploads/
```

Bestandsnamen gebruiken:

```text
YYYYMMDD-HHMMSS-full-record.mp3
YYYYMMDD-HHMMSS-Vogelnaam.wav
```

## Instellingen

Je kunt instellingen via environment variables aanpassen:

```bash
export BIRDNET_LAT=52.3702
export BIRDNET_LON=4.8952
export BIRDNET_MIN_CONFIDENCE=0.60
export BIRDNET_MODEL_VERSION=2.4
export BIRDNET_FULL_RECORD_FORMAT=mp3
export BIRDNET_FULL_RECORD_MP3_BITRATE=128k
export BIRDNET_PORT=5055
python app.py
```

`Analyzer` uit `birdnetlib` gebruikt het BirdNET-Analyzer model, niet de oude Lite-analyzer. De standaardlocatie staat op het geografisch midden van Nederland; vul je eigen coördinaten in voor betere filtering.

Voor de volledige sessie-opname kun je `BIRDNET_FULL_RECORD_FORMAT` instellen op `mp3`, `m4a`, `mp4` of `wav`. `mp4` wordt als audio-only `.m4a` opgeslagen, omdat dat voor audio-opnames het gebruikelijke MP4-containerformaat is. Detectiefragmenten blijven WAV, zodat ze zonder kwaliteitsverlies direct opnieuw geanalyseerd kunnen worden.

De webinterface toont ook:

- het gebruikte invoerapparaat;
- de tijdelijke WAV-grootte vóór compressie;
- de uiteindelijke bestandsgrootte na compressie naar MP3/M4A.
- een 100 x 100 px placeholder-afbeelding per gevonden vogel.
- een live HTML5 Canvas-spectrogram van de laatste audiobuffer;
- een PNG-spectrogram naast elke detectie.

## Spectrogrammen

De app gebruikt `scipy.signal.spectrogram` voor de berekening. De frontend tekent het live spectrogram met HTML5 Canvas via:

```text
GET /api/spectrogram/latest
```

Per detectie wordt ook een PNG opgeslagen en getoond via:

```text
/spectrograms/<bestandsnaam>.png
```

## Audiobestand uploaden

Via de webinterface kun je een bestaand audiobestand uploaden en analyseren alsof het live audio is. Ondersteunde formaten:

```text
mp3, mp4, m4a, mp4a, wav, aiff, aif
```

De uploadroute is:

```text
POST /api/upload
```

Form field:

```text
audio_file
```

De app slaat het originele bestand op in `data/uploads/`, converteert intern tijdelijk naar 48 kHz mono WAV met `ffmpeg`, analyseert in blokken van 3 seconden, en stuurt gevonden detecties via dezelfde live detectielijst als de microfoonopname.

## Nederlandse vogelnamen

`config/dutch_names.csv` bevat een uitbreidbare mapping van wetenschappelijke naam naar Nederlandse naam. Als je een volledige Nederlandse BirdNET-labelset hebt, kun je die gebruiken met:

```bash
export BIRDNET_LABELS_NL=/pad/naar/BirdNET_GLOBAL_6K_V2.4_Labels_nl.txt
python app.py
```

Zonder mapping valt de app terug op de naam die BirdNET teruggeeft.

## Problemen met microfoon

Geef Terminal of je editor toestemming voor microfoontoegang in macOS:

```text
Systeeminstellingen > Privacy en beveiliging > Microfoon
```

Bekijk beschikbare audio-apparaten via:

```text
http://127.0.0.1:5055/api/devices
```

## Handige endpoints

| Endpoint | Functie |
| --- | --- |
| `/` | Webinterface |
| `/api/status` | Huidige opname- en applicatiestatus |
| `/api/devices` | Beschikbare audioapparaten |
| `/api/start` | Live opname starten |
| `/api/stop` | Live opname stoppen |
| `/api/upload` | Audiobestand uploaden en analyseren |
| `/api/spectrogram/latest` | Laatste live spectrogram |
| `/events` | Live updates via Server-Sent Events |

## Projectstructuur

```text
app.py                   Flask-applicatie en frontend
config/dutch_names.csv   Nederlandse naamgeving
requirements.txt         Python-afhankelijkheden
data/                    lokale runtime-output, niet versioneren
```

`data/`, `.venv/`, `.env`-bestanden en lokale macOS-bestanden zijn uitgesloten via `.gitignore`. Opnames en uploads worden dus niet naar GitHub gepusht.

## Licentie en externe modellen

Deze repository bevat de applicatiecode. BirdNET, TensorFlow en de overige dependencies hebben hun eigen licenties en voorwaarden; controleer die voordat je de applicatie verder verspreidt of commercieel gebruikt.

## Credits en gebruikte software

De applicatiecode en integratie zijn gemaakt door [Dirk Boersma](https://github.com/dirkboersma).

Met dank aan de makers en maintainers van de volgende projecten:

- [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) — het model en de analysetechnologie voor vogelgeluiden.
- [birdnetlib](https://github.com/joeweiss/birdnetlib) — Python-interface voor BirdNET-analyse.
- [Flask](https://flask.palletsprojects.com/) en [Werkzeug](https://werkzeug.palletsprojects.com/) — de lokale webserver, routing en veilige bestandsnamen.
- [NumPy](https://numpy.org/) — verwerking van audiosamples en numerieke arrays.
- [SciPy](https://scipy.org/) — berekening van spectrogrammen.
- [Matplotlib](https://matplotlib.org/) — genereren van PNG-spectrogrammen.
- [python-sounddevice](https://python-sounddevice.readthedocs.io/) — toegang tot de microfoon en audioapparaten.
- [librosa](https://librosa.org/) en [TensorFlow](https://www.tensorflow.org/) — audio-/modelondersteuning binnen de BirdNET-stack.
- [FFmpeg](https://ffmpeg.org/) — conversie van geüploade bestanden en compressie van sessie-opnames.
- [uv](https://docs.astral.sh/uv/) — Python-versiebeheer, virtuele omgeving en dependency-installatie.

Controleer voor distributie of commercieel gebruik altijd de actuele licentievoorwaarden van deze externe projecten en de bijbehorende BirdNET-modelbestanden.
