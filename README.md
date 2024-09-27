Acutis converts PDF documents into html files; it does this by using CRAFT, OpenCV, Surya, and Google Lens. 

# Installation

1. Clone the repository.
```cmd
git clone https://github.com/2OsZI4ISYd/acutis.git
```
2. Install the node packages.
```cmd
npm install
```
3. Install the python requirements.
```cmd
pip install -r requirements.txt
```

# Usage

There are three modes: normal mode, replacement mode, and dataset mode. 

- Normal mode requries an input directory and an output directory and will place the html file results in the output directory. 
- Replacement mode will only take an input directory; it will replace the initial PDFs with the html files as it works. This is ideal for archival purposes. 
- Dataset mode works identically to replacement mode, but will replace the input PDFs with a page-wise splitted collection of .html files, along with their corresponding .png files; this output dataset structure can be used for training document understanding models. 

## Normal Mode

```cmd
python start.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR --chunk_size CHUNK_SIZE
```
- `INPUT_DIR`: The input directory that contains PDFs
- `OUTPUT_DIR`: The output directory where the resulting text files will be placed
- `CHUNK_SIZE`: The number of pages that will be processed at a time; the default is 10. Increase or decrease based on your memory resources.

## Replacement Mode

```cmd
python start.py --input_dir INPUT_DIR --replace --chunk_size CHUNK_SIZE
```
- `INPUT_DIR` and `CHUNK_SIZE` are used identically to how they are used in normal mode.
- `--replace` flag is added to ensure intentional usage of replacement mode.

## Dataset Mode

```cmd
python start.py --input_dir INPUT_DIR --dataset --chunk_size CHUNK_SIZE
```