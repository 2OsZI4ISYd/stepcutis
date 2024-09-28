stepcutis is a document analysis program. It uses:
- [CRAFT] to generate a synthetic document of each page of each document, effectively "denoising" the visual information on each page. This allows for a superior parsing of text-dense pages and regions in general, especially for messy-scanned documents and historical documents
- [surya] to parse the layout and find the reading order of the layout regions
- [GOT-OCR2.0] for region-to-text, allowing for accurate transcription of English and Chinese characters, along with Tex-like transcription of formulas, diagrams, figures, and music notes

## Prerequisites
1. Linux
2. CUDA >= 12.1
3. Conda

## Setup
1. Clone and enter the repository.
```bash
git clone https://github.com/2OsZI4ISYd/stepcutis.git && cd stepcutis
```
2. Make the setup script executable.
```bash
chmod +x setup.sh
```
3. Run the setup script.
```bash
./setup.sh
```
## Usage
`stepcutis` takes an input directory from the user, finds all of the PDFs in that directory and its subdirectories, converts them to html files, and places those converted files in the same directory as the original PDF files, with the same file name. To run:
```bash
stepcutis INPUT_DIR CHUNK_SIZE
```
- `INPUT_DIR`: the input directory
- `CHUNK_SIZE`: the number of pages of each PDF to process at a time; increase based on your resources.
-----
If you'd like to uninstall stepcutis, then run
```bash
stepcutis uninstall
```

