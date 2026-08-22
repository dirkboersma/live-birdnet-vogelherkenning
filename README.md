# Live BirdNET vogelherkenning

Een Flask-webapp voor het herkennen van vogelgeluiden via een browsermicrofoon, een lokale microfoon of een geüpload audiobestand. De app gebruikt [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer) voor de analyse en toont detecties, confidence-scores, audiofragmenten en spectrogrammen in een eenvoudige webinterface.

De applicatie is bedoeld voor onderzoek en kleinschalig gebruik. Een detectie is een modelvoorspelling en geen gegarandeerde soortbepaling.

## Mogelijkheden

- Browsermicrofoon live analyseren in blokken van drie seconden.
- Vogelgeluiden filteren op geografische locatie en minimale confidence.
- De volledige sessie lokaal opslaan als MP3, M4A of WAV.
- Detectiefragmenten en spectrogrammen lokaal bewaren.
- Bestaande audiobestanden uploaden voor analyse.
- Nederlandse vogelnamen tonen via een uitbreidbare mapping.

## Installatie

Gebruik bij voorkeur Python 3.11: dat is de versie waarop deze applicatie is getest. De huidige TensorFlow-versies ondersteunen ook Python 3.13; `audioop-lts` uit `requirements.txt` vult daar de uit Python verwijderde audiomodule aan. Python 3.14 wordt momenteel niet ondersteund door deze BirdNET/TensorFlow-stack.

Vereist zijn daarnaast `uv`, `portaudio` en `ffmpeg`. Installeer deze systeemafhankelijkheden via de pakketbeheerder of installer van je besturingssysteem. De eerste installatie en de eerste analyse kunnen extra modelbestanden downloaden.

Installeer `portaudio` en `ffmpeg` met de voor jouw besturingssysteem geschikte pakketbeheerder. Maak daarna de virtuele omgeving aan:

```bash
uv python install 3.11
uv venv --python 3.11
```

Activeer de omgeving volgens je platform. Gebruik bijvoorbeeld op Unix-systemen:

```bash
source .venv/bin/activate
```

Gebruik in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Installeer daarna de Python-afhankelijkheden:

```bash
python -m pip install -r requirements.txt
```

Activeer de virtuele omgeving zoals hierboven beschreven en start daarna de app:

```bash
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

## Browsermicrofoon en online gebruik

De primaire modus gebruikt de microfoon van de bezoeker in de browser. Elke drie seconden stuurt de browser een compact fragment naar de server: WebM/Opus in Chrome en Firefox waar beschikbaar, met een Safari-geschikte MP4/AAC-terugval. De server zet deze formaten tijdelijk naar 48 kHz mono WAV om voor BirdNET. Gebruik hiervoor systeem-`ffmpeg`; als dat niet beschikbaar is, installeert `imageio-ffmpeg` uit `requirements.txt` een gebruikersruimte-alternatief. De tijdelijke browserfragmenten worden na analyse verwijderd; alleen detectieclips en spectrogrammen blijven bewaard.

Een browsermicrofoon vereist HTTPS in productie. `localhost` is geschikt voor lokaal testen. De server verwerkt hoogstens twee nog wachtende fragmenten; bij overbelasting slaat de app een live fragment over zodat de resultaten actueel blijven.

### Server starten

Gebruik op een server één Gunicorn-worker; BirdNET laadt zijn model in het geheugen en meerdere workers zouden dat model elk afzonderlijk laden. Installeer ook `ffmpeg` op de server en laat Nginx HTTPS afhandelen en doorsturen naar Gunicorn op `127.0.0.1:5055`.

```bash
python3.11 -m venv .venv  # gebruik python3.13 als 3.11 niet beschikbaar is
source .venv/bin/activate
python -m pip install -r requirements.txt
gunicorn --workers 1 --threads 4 --bind 127.0.0.1:5055 wsgi:app
```

Zet dit Gunicorn-commando vervolgens onder de process-manager van de hostingpartij. Publiceer de app alleen achter een HTTPS-domein; de browser vraagt daar bij de eerste start om microfoontoestemming.

## Problemen met lokale microfoon

Geef de gebruikte terminal, editor of applicatie toestemming voor microfoontoegang in de privacy-instellingen van je besturingssysteem:

```text
Privacy-instellingen > Microfoon
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
| `/api/live-chunk` | Compact browserfragment uploaden voor live analyse |
| `/api/spectrogram/latest` | Laatste live spectrogram |
| `/events` | Live updates via Server-Sent Events |

## Projectstructuur

```text
app.py                   Flask-applicatie en frontend
config/dutch_names.csv   Nederlandse naamgeving
requirements.txt         Python-afhankelijkheden
data/                    lokale runtime-output, niet versioneren
```

`data/`, `.venv/`, `.env`-bestanden en lokale besturingssysteembestanden zijn uitgesloten via `.gitignore`. Opnames en uploads worden dus niet naar GitHub gepusht.

## Licentie en externe modellen

Deze repository bevat de applicatiecode. BirdNET, TensorFlow en de overige dependencies hebben hun eigen licenties en voorwaarden; controleer die voordat je de applicatie verder verspreidt of commercieel gebruikt.

## Credits en gebruikte software

Deze applicatie combineert de onderstaande open-sourceprojecten en libraries:

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
- [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) — gebruikersruimte-terugval voor FFmpeg op servers zonder systeembinary.
- [audioop-lts](https://pypi.org/project/audioop-lts/) en [audioread](https://github.com/beetbox/audioread) — audiocompatibiliteit voor Python 3.13 en de BirdNET-stack.
- [uv](https://docs.astral.sh/uv/) — Python-versiebeheer, virtuele omgeving en dependency-installatie.

Controleer voor distributie of commercieel gebruik altijd de actuele licentievoorwaarden van deze externe projecten en de bijbehorende BirdNET-modelbestanden.
