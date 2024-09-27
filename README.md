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
`stepcutis` works by taking a directory that the user points it to, finding all of the PDFs in that directory and its subdirectories, converting them to html files, and placing those converted files in the same directory as the original PDF files, with the same file name. To run:
```bash
conda run -n stepcutis python start.py --input_dir INPUT_DIR --chunk_size CHUNK_SIZE
```
- `INPUT_DIR`: the input directory
- `CHUNK_SIZE`: the number of pages of each PDF to process at a time; increase based on your resources.
