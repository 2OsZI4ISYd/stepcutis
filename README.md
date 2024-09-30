stepcutis is a document analysis program. It uses:
- [boomb0om](https://github.com/boomb0om/CRAFT-text-detection)'s implementation of [CRAFT](https://github.com/clovaai/CRAFT-pytorch) to generate a synthetic document of each page of each document, effectively "denoising" the visual information on each page. This allows for a superior parsing of text-dense pages and regions in general, especially for messy-scanned documents and historical documents
- Vik Paruchuri's [surya](https://github.com/VikParuchuri/surya) to parse the layout and find the reading order of the layout regions
- [stepfun](https://www.stepfun.com/)'s [GOT-OCR2.0](https://github.com/VikParuchuri/surya) for region-to-text, allowing for accurate transcription of English and Chinese characters, along with Tex-like transcription of formulas, diagrams, figures, and music notes

## Prerequisites
1. Linux
2. CUDA >= 12.1
3. Conda

## Setup
In a terminal:
```bash
curl -sSL https://github.com/2OsZI4ISYd/stepcutis/raw/refs/heads/master/grab.sh | bash
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

